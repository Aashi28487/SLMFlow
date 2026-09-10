"""JSONL tracing. Stdlib ``logging`` only, and no ``print`` anywhere.

Every run emits one JSON object per line so a trace file is greppable, appendable
and safe to truncate. The schema is the serialised form of
:class:`slmflow.types.RunTrace`, with ``schema_version`` so later analysis code
can detect a format change instead of silently misreading old runs.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from slmflow.types import Generation, RunTrace, StepTrace

__all__ = [
    "SCHEMA_VERSION",
    "JSONLHandler",
    "TraceWriter",
    "get_logger",
    "trace_to_dict",
    "read_traces",
    "new_run_id",
]

SCHEMA_VERSION = 1

_LOGGER_NAME = "slmflow"


def get_logger(name: str = "") -> logging.Logger:
    """Return a library logger.

    The library never configures the root logger and never adds a handler to it;
    an application opts in by attaching :class:`JSONLHandler` or its own.
    """
    logger = logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)
    logger.addHandler(logging.NullHandler())
    return logger


def new_run_id() -> str:
    """A short, sortable-enough run identifier."""
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _jsonable(value: Any) -> Any:
    """Coerce dataclasses, enums, tuples and paths into JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def trace_to_dict(trace: RunTrace) -> dict[str, Any]:
    """Serialise a run trace, adding the derived cost aggregates.

    The aggregates are stored as well as computed so a results table can be
    rebuilt from the JSONL alone, without importing this library.
    """
    payload = _jsonable(trace)
    assert isinstance(payload, dict)  # noqa: S101 - narrowing for type checkers
    payload["schema_version"] = SCHEMA_VERSION
    payload["derived"] = {
        "total_swaps": trace.total_swaps,
        "swap_seconds": round(trace.swap_seconds, 4),
        "total_generations": trace.total_generations,
        "output_tokens": trace.output_tokens,
        "peak_vram_bytes": trace.peak_vram_bytes,
        "peak_ram_bytes": trace.peak_ram_bytes,
        "models_used": list(trace.models_used),
        "wall_s": round(trace.wall_s, 4),
    }
    return payload


class JSONLHandler(logging.Handler):
    """A ``logging`` handler that writes one JSON object per record.

    Structured fields are passed via ``extra={"slmflow": {...}}``; anything else
    is recorded as a plain message so ordinary log lines survive too.
    """

    def __init__(self, path: str | os.PathLike[str], *, level: int = logging.INFO) -> None:
        """Open ``path`` for appending, creating parent directories."""
        super().__init__(level=level)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        """Append one JSON line. Never raises into the caller."""
        try:
            payload: dict[str, Any] = {
                "ts": round(record.created, 4),
                "level": record.levelname,
                "logger": record.name,
                "event": getattr(record, "event", record.getMessage()),
            }
            extra = getattr(record, "slmflow", None)
            if isinstance(extra, Mapping):
                payload.update(_jsonable(dict(extra)))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 - logging must never break a run
            self.handleError(record)


class TraceWriter:
    """Accumulates step traces for one run and writes the record on close.

    Use as a context manager so a crashed run still flushes what it had:

        >>> from slmflow.telemetry.trace import TraceWriter
        >>> with TraceWriter("runs/demo.jsonl", pipeline="demo", device="t1") as w:
        ...     pass
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None,
        *,
        pipeline: str,
        device: str,
        task_id: str = "",
        seed: int | None = None,
        config_digest: str = "",
    ) -> None:
        """Start a run record. ``path=None`` keeps the trace in memory only."""
        self.path = Path(path) if path is not None else None
        self._started = time.perf_counter()
        self._log = get_logger("telemetry")
        self.trace = RunTrace(
            run_id=new_run_id(),
            task_id=task_id,
            pipeline=pipeline,
            device=device,
            seed=seed,
            config_digest=config_digest,
        )

    def add_step(self, step: StepTrace) -> None:
        """Append a step to the run record."""
        self.trace = self.trace.with_step(step)
        self._log.info(
            "step",
            extra={
                "event": "step",
                "slmflow": {
                    "run_id": self.trace.run_id,
                    "step": step.step,
                    "model": step.model_id,
                    "strategy": step.strategy,
                    "n_generations": step.n_generations,
                    "confidence": step.confidence,
                    "swaps": step.swaps,
                    "swap_seconds": round(step.swap_seconds, 4),
                    "abstained": step.abstained,
                    "wall_s": round(step.wall_s, 4),
                },
            },
        )

    def finish(self) -> RunTrace:
        """Stamp the wall time, write the record, and return the final trace."""
        from dataclasses import replace

        self.trace = replace(self.trace, wall_s=time.perf_counter() - self._started)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(trace_to_dict(self.trace), ensure_ascii=False) + "\n")
        return self.trace

    def __enter__(self) -> TraceWriter:
        """Enter the run scope."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Always write the record, even when the run raised."""
        self.finish()


@contextmanager
def _timed() -> Iterator[list[float]]:
    """Yield a one-element list that receives the elapsed seconds on exit."""
    box: list[float] = []
    start = time.perf_counter()
    try:
        yield box
    finally:
        box.append(time.perf_counter() - start)


def read_traces(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Lazily read a JSONL trace file, skipping malformed lines.

    A truncated final line is common when a run was killed, so it is skipped
    rather than treated as corruption of the whole file.
    """
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def generation_summary(generations: tuple[Generation, ...]) -> dict[str, Any]:
    """Compact per-step cost summary used by the run inspector."""
    if not generations:
        return {"n": 0}
    return {
        "n": len(generations),
        "output_tokens": sum(g.output_tokens for g in generations),
        "latency_s": round(sum(g.latency_s for g in generations), 4),
        "load_s": round(sum(g.load_s for g in generations), 4),
        "cold": sum(1 for g in generations if g.was_cold),
    }
