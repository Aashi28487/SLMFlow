"""Ollama backend — talks to the local Ollama REST API.

Ollama exposes a simple HTTP API (default ``http://localhost:11434``).
This backend treats the Ollama process as the residency owner: ``load``
tells Ollama to keep the model warm, ``unload`` sets ``keep_alive=0``
so Ollama may release it.

No binary dependency; only ``urllib`` from the stdlib.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from slmflow.types import Generation, MemoryFootprint, ModelCard

__all__ = ["OllamaBackend"]

_DEFAULT_BASE = "http://localhost:11434"


class OllamaBackend:
    """Backend that delegates to a running ``ollama serve`` process."""

    name = "ollama"

    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self._base = (base_url or os.environ.get("OLLAMA_HOST") or _DEFAULT_BASE).rstrip("/")
        self._timeout = timeout
        self._loaded: set[str] = set()

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        """True when the Ollama daemon is reachable."""
        try:
            req = urllib.request.Request(f"{self._base}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    # ------------------------------------------------------------------ #
    # Residency
    # ------------------------------------------------------------------ #
    def load(self, card: ModelCard) -> None:
        """Warm-load the model by sending a prompt with keep_alive=-1."""
        if card.model_id in self._loaded:
            return
        body = json.dumps({
            "model": card.model_id,
            "prompt": "",
            "keep_alive": -1,  # keep resident indefinitely
        }).encode()
        req = urllib.request.Request(
            f"{self._base}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                # Consume the streaming response
                for line in resp:
                    pass
            self._loaded.add(card.model_id)
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Failed to load {card.model_id} via Ollama: {e}") from e

    def unload(self, card: ModelCard) -> None:
        """Tell Ollama to unload the model by setting keep_alive=0."""
        body = json.dumps({
            "model": card.model_id,
            "prompt": "",
            "keep_alive": 0,
        }).encode()
        req = urllib.request.Request(
            f"{self._base}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                for line in resp:
                    pass
        except (urllib.error.URLError, OSError):
            pass
        self._loaded.discard(card.model_id)

    def is_loaded(self, card: ModelCard) -> bool:
        """Check whether the model is currently loaded in Ollama."""
        try:
            req = urllib.request.Request(f"{self._base}/api/ps", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            for model_info in data.get("models", []):
                if model_info.get("name", "").startswith(card.model_id):
                    return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass
        return card.model_id in self._loaded

    # ------------------------------------------------------------------ #
    # Measurement
    # ------------------------------------------------------------------ #
    def measure_footprint(self, card: ModelCard) -> MemoryFootprint:
        """Measure the actual memory used by the model via /api/ps."""
        vram_bytes = 0
        ram_bytes = 0
        try:
            req = urllib.request.Request(f"{self._base}/api/ps", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            for model_info in data.get("models", []):
                if model_info.get("name", "").startswith(card.model_id):
                    size = model_info.get("size", 0)
                    size_vram = model_info.get("size_vram", 0)
                    # size_vram is VRAM portion, remainder is RAM
                    vram_bytes = size_vram
                    ram_bytes = max(0, size - size_vram)
                    break
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            # Fallback: try nvidia-smi for VRAM
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    vram_mb = int(result.stdout.strip().split("\n")[0])
                    vram_bytes = vram_mb * 1024 * 1024
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass
        return MemoryFootprint(vram_bytes=vram_bytes, ram_bytes=ram_bytes)

    def digest_for(self, model_id: str) -> str | None:
        """Get the model digest from Ollama's model list."""
        try:
            req = urllib.request.Request(f"{self._base}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            for model in data.get("models", []):
                if model.get("name", "").startswith(model_id):
                    return model.get("digest", "")
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass
        return None

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
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
        """Generate a completion using Ollama's /api/generate endpoint."""
        options: dict[str, Any] = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            options["seed"] = seed

        payload: dict[str, Any] = {
            "model": card.model_id,
            "prompt": prompt,
            "stream": False,
            "options": options,
            "keep_alive": -1,
        }
        if stop:
            payload["options"]["stop"] = list(stop)
        # Ollama cannot take a GBNF grammar, but it can constrain output to
        # valid JSON. The strategy layer appends "Reply with JSON only" exactly
        # when the step has an output schema, so that marker enables JSON mode.
        if grammar is not None or "Reply with JSON only" in prompt:
            payload["format"] = "json"

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._base}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                result = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as e:
            latency = time.perf_counter() - t0
            return Generation(
                text=f"ERROR: {e}",
                model_id=card.model_id,
                prompt_tokens=0,
                output_tokens=0,
                latency_s=latency,
            )
        latency = time.perf_counter() - t0

        text = result.get("response", "")
        prompt_tokens = result.get("prompt_eval_count", 0) or 0
        output_tokens = result.get("eval_count", 0) or 0

        return Generation(
            text=text.strip(),
            model_id=card.model_id,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            load_s=0.0,
            was_cold=False,
            schema_valid=None,
        )
