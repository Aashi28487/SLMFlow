"""``llama-cpp-python`` backend: in-process inference with GPU offload.

Optional extra ``slmflow[llama]``. The import of ``llama_cpp`` happens inside
:meth:`LlamaCppBackend.load`, never at module import, so ``import slmflow``
stays dependency-free on T0.

Memory accounting here is **in-process**: the weights live in this process's
address space, so RSS delta is the right measurement. VRAM is read from
``nvidia-smi`` because llama.cpp does not report its own allocation. Both are
measured, never derived from the file size.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slmflow.errors import BackendUnavailable
from slmflow.resource.rss import gpu_used_bytes, process_rss_bytes
from slmflow.telemetry.trace import get_logger
from slmflow.types import Generation, MemoryFootprint, ModelCard

__all__ = ["LlamaCppBackend"]

_LOG = get_logger("backend.llama_cpp")


class LlamaCppBackend:
    """Runs GGUF models in this process via ``llama-cpp-python``.

    Args:
        model_dir: Directory holding the ``.gguf`` files. A card's
            ``model_id`` is resolved as ``model_dir / f"{model_id}.gguf"``
            unless ``paths`` supplies an explicit path.
        paths: Explicit ``{model_id: path}`` overrides.
        n_gpu_layers: Layers to offload. ``-1`` offloads everything, which is
            what a 7-8B Q4_K_M on 6 GB VRAM wants. ``0`` forces pure CPU.
        n_ctx: Context window. Larger contexts cost VRAM, so this is part of
            the memory budget, not a free parameter.
        n_threads: CPU threads. Defaults to llama.cpp's own choice.
        verbose: Pass through to llama.cpp's own logging.
    """

    name = "llama_cpp"

    def __init__(
        self,
        model_dir: str | os.PathLike[str] = "models",
        *,
        paths: dict[str, str] | None = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Store configuration. No import and no allocation happens here."""
        self.model_dir = Path(model_dir)
        self.paths = dict(paths or {})
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self._handles: dict[str, Any] = {}
        self._baseline_rss: dict[str, int] = {}
        self._baseline_vram: dict[str, int] = {}

    # -- Backend protocol ------------------------------------------------- #
    def is_available(self) -> bool:
        """True when ``llama_cpp`` imports on this machine."""
        try:
            import llama_cpp  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def path_for(self, card: ModelCard) -> Path:
        """Resolve the GGUF path for a card.

        Raises:
            BackendUnavailable: When the weights are not on disk.
        """
        explicit = self.paths.get(card.model_id)
        path = Path(explicit) if explicit else self.model_dir / f"{card.model_id}.gguf"
        if not path.exists():
            raise BackendUnavailable(
                f"GGUF weights not found at {path}",
                backend=self.name,
                install_hint=(
                    f"download the model into {self.model_dir}/ as {card.model_id}.gguf, "
                    f"or pass paths={{'{card.model_id}': '<path>'}}"
                ),
            )
        return path

    def load(self, card: ModelCard) -> None:
        """Instantiate the model. Idempotent.

        Raises:
            BackendUnavailable: When the extra is not installed or the weights
                are missing.
        """
        if card.model_id in self._handles:
            return
        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailable(
                "llama-cpp-python is not installed",
                backend=self.name,
                install_hint="pip install 'slmflow[llama]'",
            ) from exc

        path = self.path_for(card)
        self._baseline_rss[card.model_id] = process_rss_bytes()
        self._baseline_vram[card.model_id] = gpu_used_bytes()

        kwargs: dict[str, Any] = {
            "model_path": str(path),
            "n_ctx": card.context_tokens or self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "logits_all": False,
            "verbose": self.verbose,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads

        start = time.perf_counter()
        self._handles[card.model_id] = Llama(**kwargs)
        elapsed = time.perf_counter() - start
        _LOG.info(
            "llama.cpp model loaded",
            extra={
                "event": "backend_load",
                "slmflow": {
                    "model": card.model_id,
                    "path": str(path),
                    "n_gpu_layers": self.n_gpu_layers,
                    "n_ctx": kwargs["n_ctx"],
                    "load_s": round(elapsed, 4),
                },
            },
        )

    def unload(self, card: ModelCard) -> None:
        """Drop the handle and let the allocator reclaim. Idempotent.

        llama.cpp frees on ``__del__``, so the handle is closed explicitly
        where the binding supports it before dropping the reference.
        """
        handle = self._handles.pop(card.model_id, None)
        if handle is None:
            return
        close = getattr(handle, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - best effort
                _LOG.warning(
                    "llama.cpp close failed",
                    extra={
                        "event": "backend_close_failed",
                        "slmflow": {"model": card.model_id, "error": repr(exc)},
                    },
                )
        del handle
        self._baseline_rss.pop(card.model_id, None)
        self._baseline_vram.pop(card.model_id, None)

    def is_loaded(self, card: ModelCard) -> bool:
        """True when a handle is held."""
        return card.model_id in self._handles

    def measure_footprint(self, card: ModelCard) -> MemoryFootprint:
        """Measure real bytes: RSS delta for CPU, nvidia-smi delta for VRAM.

        Weights are in this process, so RSS delta since just before load is the
        honest CPU number. When layers are offloaded, the VRAM delta is charged
        to the VRAM tier and the RAM delta shrinks accordingly -- which is
        exactly the two-tier accounting C1 needs.
        """
        if card.model_id not in self._handles:
            return MemoryFootprint()
        ram_delta = max(0, process_rss_bytes() - self._baseline_rss.get(card.model_id, 0))
        vram_delta = max(0, gpu_used_bytes() - self._baseline_vram.get(card.model_id, 0))
        disk = 0
        try:
            disk = self.path_for(card).stat().st_size
        except (BackendUnavailable, OSError):
            pass
        return MemoryFootprint(vram_bytes=vram_delta, ram_bytes=ram_delta, disk_bytes=disk)

    def generate(
        self,
        card: ModelCard,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
        grammar: str | None = None,
        stop: Sequence[str] = (),
    ) -> Generation:
        """Complete a prompt, optionally under a GBNF grammar.

        A grammar makes malformed structured output impossible rather than
        unlikely, which is the Phase 10 schema-validity mechanism.

        Raises:
            BackendUnavailable: When the model was never loaded.
        """
        handle = self._handles.get(card.model_id)
        if handle is None:
            raise BackendUnavailable(
                f"Model {card.model_id!r} is not loaded",
                backend=self.name,
                install_hint="wrap the call in ResourceManager.resident()",
            )

        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": list(stop),
            "logprobs": 1 if card.supports_logprobs else None,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if grammar:
            kwargs["grammar"] = self._compile_grammar(grammar)

        start = time.perf_counter()
        raw = handle(prompt, **{k: v for k, v in kwargs.items() if v is not None})
        latency = time.perf_counter() - start

        choice = raw["choices"][0]
        usage = raw.get("usage", {})
        logprob_sum, margin = _logprob_stats(choice.get("logprobs"))
        return Generation(
            text=str(choice.get("text", "")).strip(),
            model_id=card.model_id,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=latency,
            load_s=0.0,
            was_cold=False,
            logprob_sum=logprob_sum,
            top_token_margin=margin,
            finish_reason=str(choice.get("finish_reason", "stop")),
        )

    @staticmethod
    def _compile_grammar(gbnf: str) -> Any:
        """Compile a GBNF string into a llama.cpp grammar object.

        Raises:
            BackendUnavailable: When the binding lacks grammar support.
        """
        try:
            from llama_cpp import LlamaGrammar  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - very old bindings
            raise BackendUnavailable(
                "This llama-cpp-python build has no LlamaGrammar",
                backend="llama_cpp",
                install_hint="pip install -U 'llama-cpp-python>=0.3.2'",
            ) from exc
        return LlamaGrammar.from_string(gbnf, verbose=False)


def _logprob_stats(logprobs: Any) -> tuple[float | None, float | None]:
    """Extract the log-prob sum and the mean top-2 margin from a response.

    The margin is a Phase 10 confidence signal that does not require sampling
    ``k`` times, so it is the cheapest signal available on CPU.
    """
    if not isinstance(logprobs, dict):
        return None, None
    token_logprobs = logprobs.get("token_logprobs") or []
    values = [v for v in token_logprobs if isinstance(v, (int, float))]
    total = float(sum(values)) if values else None

    margins: list[float] = []
    for entry in logprobs.get("top_logprobs") or []:
        if isinstance(entry, dict) and len(entry) >= 2:
            ranked = sorted(entry.values(), reverse=True)
            margins.append(float(ranked[0] - ranked[1]))
    margin = sum(margins) / len(margins) if margins else None
    return total, margin
