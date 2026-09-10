"""Model residency as a context manager, with swap accounting.

``with rm.resident(card) as handle:`` is the correct API shape for something
that must be released deterministically (``BUILD_BRIEF.md`` section 3). The
manager owns every load and unload, charges swap cost, and enforces the
per-tier budget invariant. Nothing else in the library may call
``backend.load`` directly.

Eviction policy is pluggable because "which model to evict" is exactly the
question C1 studies. :class:`LRUPolicy` is the naive baseline the swap-aware
scheduler is measured against.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from slmflow.errors import ResourceExhausted
from slmflow.resource.budget import RESIDENT_TIERS, MemoryBudget, Placement
from slmflow.telemetry.trace import get_logger
from slmflow.types import Backend, MemoryFootprint, ModelCard

__all__ = [
    "EvictionPolicy",
    "LRUPolicy",
    "LargestFirstPolicy",
    "ResidencyHandle",
    "ResourceManager",
    "SwapLedger",
]

_LOG = get_logger("resource")


@dataclass(slots=True)
class SwapLedger:
    """Running tally of what residency actually cost.

    ``load_seconds`` is measured wall-clock, never predicted. This is the
    number the thesis argues every other framework hides.
    """

    loads: int = 0
    evictions: int = 0
    load_seconds: float = 0.0
    cold_loads: int = 0
    warm_hits: int = 0
    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0
    per_model_loads: dict[str, int] = field(default_factory=dict)

    def record_load(self, model_id: str, seconds: float, *, cold: bool) -> None:
        """Charge one load."""
        self.loads += 1
        self.load_seconds += seconds
        if cold:
            self.cold_loads += 1
        self.per_model_loads[model_id] = self.per_model_loads.get(model_id, 0) + 1

    def record_hit(self) -> None:
        """Charge one warm hit (no load needed)."""
        self.warm_hits += 1

    def record_eviction(self) -> None:
        """Charge one eviction."""
        self.evictions += 1

    def observe_peak(self, footprint: MemoryFootprint) -> None:
        """Update the high-water marks after a residency change."""
        self.peak_vram_bytes = max(self.peak_vram_bytes, footprint.vram_bytes)
        self.peak_ram_bytes = max(self.peak_ram_bytes, footprint.ram_bytes)

    @property
    def hit_rate(self) -> float | None:
        """Warm-hit rate, or ``None`` when nothing was requested yet."""
        total = self.loads + self.warm_hits
        return self.warm_hits / total if total else None

    def snapshot(self) -> dict[str, float | int | None]:
        """Flat dict for a trace record."""
        return {
            "loads": self.loads,
            "evictions": self.evictions,
            "load_seconds": round(self.load_seconds, 4),
            "cold_loads": self.cold_loads,
            "warm_hits": self.warm_hits,
            "hit_rate": self.hit_rate,
            "peak_vram_bytes": self.peak_vram_bytes,
            "peak_ram_bytes": self.peak_ram_bytes,
        }


class EvictionPolicy(ABC):
    """Chooses which resident models to evict to make room.

    ``ABC`` rather than ``Protocol``: subclasses share :meth:`enough`, so there
    is genuine shared implementation to inherit (see ``ARCHITECTURE.md``).
    """

    name: str = "abstract"

    @abstractmethod
    def order(self, resident: Sequence[str], last_used: dict[str, float]) -> list[str]:
        """Return resident model ids in eviction-preference order."""

    def enough(
        self,
        need: MemoryFootprint,
        budget: MemoryBudget,
        placements: dict[str, Placement],
        last_used: dict[str, float],
        *,
        protected: frozenset[str] = frozenset(),
    ) -> list[str]:
        """Pick the fewest models whose removal admits ``need``.

        Args:
            need: Footprint to make room for.
            budget: The active per-tier budget.
            placements: Currently resident models.
            last_used: Monotonic timestamps of last use.
            protected: Models that must not be evicted (in use right now).

        Returns:
            Model ids to evict, in the order to evict them. Empty if ``need``
            already fits.

        Raises:
            ResourceExhausted: If no subset of evictable models can make room.
        """
        if budget.admits(need, placements.values()):
            return []
        candidates = [m for m in self.order(list(placements), last_used) if m not in protected]
        chosen: list[str] = []
        remaining = dict(placements)
        for model_id in candidates:
            chosen.append(model_id)
            remaining.pop(model_id, None)
            if budget.admits(need, remaining.values()):
                return chosen
        # Nothing left to give: report the tightest tier.
        free = budget.headroom(remaining.values())
        worst = max(RESIDENT_TIERS, key=lambda t: need.get(t) - free.get(t))
        raise ResourceExhausted(
            "No eviction set frees enough memory",
            tier=str(worst),
            requested_bytes=need.get(worst),
            available_bytes=free.get(worst),
            budget_bytes=budget.get(worst),
        )


class LRUPolicy(EvictionPolicy):
    """Least-recently-used. The naive baseline the C1 ablation compares against."""

    name = "lru"

    def order(self, resident: Sequence[str], last_used: dict[str, float]) -> list[str]:
        """Oldest use first."""
        return sorted(resident, key=lambda m: last_used.get(m, 0.0))


class LargestFirstPolicy(EvictionPolicy):
    """Evict the biggest model first, to minimise the number of evictions.

    Requires a footprint lookup, injected at construction, because size is not
    derivable from the model id alone.
    """

    name = "largest_first"

    def __init__(self, footprints: dict[str, MemoryFootprint]) -> None:
        """Store the footprint table this policy sorts on."""
        self._footprints = footprints

    def order(self, resident: Sequence[str], last_used: dict[str, float]) -> list[str]:
        """Largest resident bytes first, breaking ties by least-recent use."""
        return sorted(
            resident,
            key=lambda m: (
                -self._footprints.get(m, MemoryFootprint()).resident_bytes,
                last_used.get(m, 0.0),
            ),
        )


@dataclass(slots=True)
class ResidencyHandle:
    """Proof that a model is resident, plus what this residency cost.

    Yielded by :meth:`ResourceManager.resident`. Valid only inside the ``with``
    block; the manager may evict the model once the block exits.
    """

    card: ModelCard
    backend: Backend
    was_cold: bool
    load_seconds: float
    evicted: tuple[str, ...] = ()

    @property
    def model_id(self) -> str:
        """Convenience accessor."""
        return self.card.model_id

    def generate(self, prompt: str, **kwargs: object) -> object:
        """Generate through the backend that owns this residency."""
        return self.backend.generate(self.card, prompt, **kwargs)  # type: ignore[arg-type]


class ResourceManager:
    """Owns model residency under a hard per-tier memory budget.

    Every load is charged to a :class:`SwapLedger`; every admission is checked
    against :class:`MemoryBudget`. A budget-exceeding request either triggers a
    scheduled eviction or raises :class:`ResourceExhausted` -- it never silently
    over-commits.

    Example:
        >>> from slmflow.models.backends.mock import MockBackend
        >>> from slmflow.resource.budget import MemoryBudget
        >>> from slmflow.types import MemoryFootprint, ModelCard
        >>> card = ModelCard(
        ...     "tiny", "mock", 0.5, footprint=MemoryFootprint(ram_bytes=400_000_000)
        ... )
        >>> rm = ResourceManager(MemoryBudget.single_slot(1_000_000_000))
        >>> rm.register(card, MockBackend())
        >>> with rm.resident(card) as handle:
        ...     handle.model_id
        'tiny'
    """

    def __init__(
        self,
        budget: MemoryBudget,
        *,
        policy: EvictionPolicy | None = None,
        measure_on_load: bool = True,
    ) -> None:
        """Create a manager.

        Args:
            budget: Hard per-tier ceiling.
            policy: Eviction policy. Defaults to :class:`LRUPolicy`.
            measure_on_load: Re-measure the real footprint after each load and
                use the measurement instead of the card's declared value. Keep
                this on for benchmarking; turn it off for deterministic tests.
        """
        self.budget = budget
        self.policy = policy or LRUPolicy()
        self.measure_on_load = measure_on_load
        self.ledger = SwapLedger()
        self._backends: dict[str, Backend] = {}
        self._cards: dict[str, ModelCard] = {}
        self._placements: dict[str, Placement] = {}
        self._last_used: dict[str, float] = {}
        self._ever_loaded: set[str] = set()
        self._in_use: set[str] = set()

    # -- registration ----------------------------------------------------- #
    def register(self, card: ModelCard, backend: Backend) -> None:
        """Associate a model card with the backend that can run it."""
        self._cards[card.model_id] = card
        self._backends[card.model_id] = backend

    def card_for(self, model_id: str) -> ModelCard:
        """Look up a registered card."""
        from slmflow.errors import ModelNotFound

        try:
            return self._cards[model_id]
        except KeyError:
            raise ModelNotFound(model_id, list(self._cards)) from None

    # -- introspection ---------------------------------------------------- #
    @property
    def resident_models(self) -> tuple[str, ...]:
        """Model ids currently occupying memory, most recently used last."""
        return tuple(sorted(self._placements, key=lambda m: self._last_used.get(m, 0.0)))

    def is_resident(self, model_id: str) -> bool:
        """True when the model currently occupies memory."""
        return model_id in self._placements

    def current_footprint(self) -> MemoryFootprint:
        """Total bytes resident right now."""
        return self.budget.used(self._placements.values())

    def headroom(self) -> MemoryFootprint:
        """Free bytes per tier."""
        return self.budget.headroom(self._placements.values())

    # -- the core operation ----------------------------------------------- #
    @contextmanager
    def resident(self, card: ModelCard | str) -> Iterator[ResidencyHandle]:
        """Guarantee a model is resident for the duration of the block.

        Loads it if needed, evicting other models under the active policy.
        The model is pinned against eviction while the block is open.

        Args:
            card: A model card or a registered model id.

        Yields:
            A :class:`ResidencyHandle` recording whether the load was cold and
            what it cost.

        Raises:
            ResourceExhausted: When the model cannot fit even on an empty
                machine, or no eviction set frees enough room.
        """
        if isinstance(card, str):
            card = self.card_for(card)
        model_id = card.model_id
        backend = self._backends.get(model_id)
        if backend is None:
            from slmflow.errors import ModelNotFound

            raise ModelNotFound(model_id, list(self._backends))

        declared = card.footprint
        if declared.resident_bytes > 0 and not self.budget.fits(declared):
            worst = max(RESIDENT_TIERS, key=lambda t: declared.get(t) - self.budget.get(t))
            raise ResourceExhausted(
                f"Model {model_id!r} cannot fit even on an empty machine",
                tier=str(worst),
                requested_bytes=declared.get(worst),
                available_bytes=self.budget.get(worst),
                budget_bytes=self.budget.get(worst),
            )

        evicted: tuple[str, ...] = ()
        load_seconds = 0.0
        was_cold = False

        if self.is_resident(model_id):
            self.ledger.record_hit()
        else:
            victims = self.policy.enough(
                declared,
                self.budget,
                self._placements,
                self._last_used,
                protected=frozenset(self._in_use),
            )
            for victim in victims:
                self._unload(victim)
            evicted = tuple(victims)

            was_cold = model_id not in self._ever_loaded
            start = time.perf_counter()
            backend.load(card)
            load_seconds = time.perf_counter() - start
            self._ever_loaded.add(model_id)
            self.ledger.record_load(model_id, load_seconds, cold=was_cold)

            measured = declared
            if self.measure_on_load:
                try:
                    measured = backend.measure_footprint(card)
                except Exception as exc:  # noqa: BLE001 - measurement must not kill a run
                    _LOG.warning(
                        "footprint measurement failed",
                        extra={
                            "event": "measure_failed",
                            "slmflow": {"model": model_id, "error": repr(exc)},
                        },
                    )
                    measured = declared
            self._placements[model_id] = Placement(model_id=model_id, footprint=measured)

            # A measured footprint larger than declared can breach the budget.
            # Report it rather than pretend the invariant held.
            over = [
                t
                for t in RESIDENT_TIERS
                if self.budget.used(self._placements.values()).get(t) > self.budget.get(t)
            ]
            if over:
                tier = over[0]
                used = self.budget.used(self._placements.values()).get(tier)
                self._unload(model_id)
                raise ResourceExhausted(
                    f"Measured footprint of {model_id!r} breaks the budget "
                    f"(declared {declared.get(tier) / 1e9:.2f} GB, "
                    f"measured {measured.get(tier) / 1e9:.2f} GB)",
                    tier=str(tier),
                    requested_bytes=used,
                    available_bytes=self.budget.get(tier),
                    budget_bytes=self.budget.get(tier),
                )

            self.ledger.observe_peak(self.current_footprint())
            _LOG.info(
                "model loaded",
                extra={
                    "event": "load",
                    "slmflow": {
                        "model": model_id,
                        "cold": was_cold,
                        "load_s": round(load_seconds, 4),
                        "evicted": list(evicted),
                        "vram_bytes": measured.vram_bytes,
                        "ram_bytes": measured.ram_bytes,
                    },
                },
            )

        self._last_used[model_id] = time.monotonic()
        self._in_use.add(model_id)
        try:
            yield ResidencyHandle(
                card=card,
                backend=backend,
                was_cold=was_cold,
                load_seconds=load_seconds,
                evicted=evicted,
            )
        finally:
            self._in_use.discard(model_id)
            self._last_used[model_id] = time.monotonic()

    # -- teardown --------------------------------------------------------- #
    def _unload(self, model_id: str) -> None:
        """Release one model, charging an eviction."""
        backend = self._backends.get(model_id)
        card = self._cards.get(model_id)
        if backend is not None and card is not None:
            try:
                backend.unload(card)
            except Exception as exc:  # noqa: BLE001 - a failed unload must not wedge the run
                _LOG.warning(
                    "unload failed",
                    extra={
                        "event": "unload_failed",
                        "slmflow": {"model": model_id, "error": repr(exc)},
                    },
                )
        self._placements.pop(model_id, None)
        self.ledger.record_eviction()
        _LOG.info("model evicted", extra={"event": "evict", "slmflow": {"model": model_id}})

    def evict(self, model_id: str) -> None:
        """Evict one model on request.

        Raises:
            ResourceExhausted: If the model is pinned by an open residency block.
        """
        if model_id in self._in_use:
            raise ResourceExhausted(
                f"Model {model_id!r} is in use and cannot be evicted",
                tier=str(RESIDENT_TIERS[0]),
                requested_bytes=0,
                available_bytes=0,
                budget_bytes=0,
            )
        if self.is_resident(model_id):
            self._unload(model_id)

    def evict_all(self) -> None:
        """Release every resident model. Safe to call repeatedly."""
        for model_id in list(self._placements):
            if model_id not in self._in_use:
                self._unload(model_id)

    def reset_ledger(self) -> SwapLedger:
        """Return the current ledger and start a fresh one."""
        old = self.ledger
        self.ledger = SwapLedger()
        return old

    def with_budget(self, budget: MemoryBudget) -> ResourceManager:
        """A new manager over the same registrations but a different budget.

        Used by the offline oracle to explore counterfactual budgets without
        disturbing the live manager.
        """
        clone = ResourceManager(
            budget, policy=self.policy, measure_on_load=self.measure_on_load
        )
        clone._backends = dict(self._backends)
        clone._cards = dict(self._cards)
        return clone

    def __enter__(self) -> ResourceManager:
        """Enter a scope that releases every model on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release everything."""
        self.evict_all()

    def __repr__(self) -> str:
        """Show the budget and what is resident."""
        return (
            f"ResourceManager({self.budget.describe()}, "
            f"resident={list(self._placements)}, loads={self.ledger.loads})"
        )


def footprints_of(cards: Sequence[ModelCard]) -> dict[str, MemoryFootprint]:
    """Helper: model id to footprint, for :class:`LargestFirstPolicy`."""
    return {c.model_id: c.footprint for c in cards}


def rehydrate_card(card: ModelCard, measured: MemoryFootprint, load_s: float) -> ModelCard:
    """Return a card updated with a measurement taken at runtime.

    Cards are frozen, so profiling produces a new card rather than mutating one.
    """
    return replace(card, footprint=measured, cold_load_s=load_s)
