"""Backends package — factory for inference engines."""

from __future__ import annotations

from typing import Any

from slmflow.errors import BackendUnavailable

__all__ = ["get_backend", "available_backends"]


def get_backend(name: str, **kwargs: Any):
    """Instantiate a backend by name.

    Args:
        name: One of ``'ollama'``, ``'llama_cpp'``, ``'mock'``.
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A backend instance satisfying the :class:`Backend` protocol.

    Raises:
        BackendUnavailable: If the backend name is not recognised.
    """
    if name == "mock":
        from slmflow.models.backends.mock import MockBackend
        return MockBackend(**kwargs)
    if name == "ollama":
        from slmflow.models.backends.ollama import OllamaBackend
        return OllamaBackend(**kwargs)
    if name == "llama_cpp":
        raise BackendUnavailable(
            "llama_cpp backend is not yet implemented. Use ollama instead."
        )
    raise BackendUnavailable(f"Unknown backend {name!r}. Choose from: ollama, llama_cpp, mock.")


def available_backends() -> dict[str, bool]:
    """Return a mapping of backend name to availability status."""
    result = {}
    for name in ("ollama", "llama_cpp", "mock"):
        try:
            b = get_backend(name)
            result[name] = b.is_available()
        except BackendUnavailable:
            result[name] = False
    return result
