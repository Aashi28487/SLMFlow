"""The memory budget: a vector over tiers, not a scalar.

The T1 evaluation machine has ~6 GB VRAM above ~16 GB system RAM above an SSD
(``BUILD_BRIEF.md`` section 2, corrected 2026-09-10). A single "peak memory"
number cannot express that machine, so the budget is a mapping from
:class:`~slmflow.types.MemoryTier` to bytes and the invariant is checked per
tier. :class:`ResourceExhausted` always names which tier failed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from slmflow.errors import ResourceExhausted
from slmflow.types import DeviceCard, MemoryFootprint, MemoryTier

__all__ = ["MemoryBudget", "Placement"]

_GB = 1_000_000_000

#: Tiers that count as residency. Disk is capacity, not residency.
RESIDENT_TIERS: tuple[MemoryTier, ...] = (MemoryTier.VRAM, MemoryTier.RAM)


@dataclass(frozen=True, slots=True)
class Placement:
    """Where one model's bytes were charged.

    A model offloaded to GPU charges VRAM; the same model on CPU charges RAM.
    Partial offload charges both, which is why this is a footprint and not a
    single tier label.
    """

    model_id: str
    footprint: MemoryFootprint

    @property
    def primary_tier(self) -> MemoryTier:
        """The tier holding most of this model's bytes."""
        return (
            MemoryTier.VRAM
            if self.footprint.vram_bytes >= self.footprint.ram_bytes
            else MemoryTier.RAM
        )


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """A hard per-tier ceiling, with the arithmetic to check it.

    Immutable: an admission decision is made against a snapshot, so a budget
    cannot be mutated underneath a pending load.

    Example:
        >>> b = MemoryBudget(vram_bytes=5_000_000_000, ram_bytes=8_000_000_000)
        >>> b.fits(MemoryFootprint(vram_bytes=4_400_000_000))
        True
        >>> b.fits(MemoryFootprint(vram_bytes=6_000_000_000))
        False
    """

    vram_bytes: int = 0
    ram_bytes: int = 0
    disk_bytes: int = 0

    @classmethod
    def from_device(cls, card: DeviceCard, headroom: float = 0.75) -> MemoryBudget:
        """Derive a budget from a probed device card, leaving OS headroom."""
        alloc = card.default_budget_bytes(headroom)
        return cls(
            vram_bytes=alloc[MemoryTier.VRAM],
            ram_bytes=alloc[MemoryTier.RAM],
            disk_bytes=alloc[MemoryTier.DISK],
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[MemoryTier, int]) -> MemoryBudget:
        """Build from a tier mapping, e.g. the output of ``budget_from_env``."""
        return cls(
            vram_bytes=mapping.get(MemoryTier.VRAM, 0),
            ram_bytes=mapping.get(MemoryTier.RAM, 0),
            disk_bytes=mapping.get(MemoryTier.DISK, 0),
        )

    @classmethod
    def single_slot(cls, ram_bytes: int) -> MemoryBudget:
        """A CPU-only, RAM-only budget. The T0 shape, and the test default."""
        return cls(vram_bytes=0, ram_bytes=ram_bytes)

    def get(self, tier: MemoryTier) -> int:
        """Ceiling for one tier."""
        if tier is MemoryTier.VRAM:
            return self.vram_bytes
        if tier is MemoryTier.RAM:
            return self.ram_bytes
        return self.disk_bytes

    def fits(self, footprint: MemoryFootprint) -> bool:
        """True when a single footprint is admissible on an empty machine."""
        return all(footprint.get(t) <= self.get(t) for t in RESIDENT_TIERS)

    def used(self, placements: Iterable[Placement]) -> MemoryFootprint:
        """Sum the footprints of everything currently resident."""
        total = MemoryFootprint()
        for placement in placements:
            total = total + placement.footprint
        return total

    def headroom(self, placements: Iterable[Placement]) -> MemoryFootprint:
        """Free bytes per tier, floored at zero."""
        used = self.used(placements)
        return MemoryFootprint(
            vram_bytes=max(0, self.vram_bytes - used.vram_bytes),
            ram_bytes=max(0, self.ram_bytes - used.ram_bytes),
            disk_bytes=max(0, self.disk_bytes - used.disk_bytes),
        )

    def admits(self, footprint: MemoryFootprint, placements: Iterable[Placement]) -> bool:
        """True when ``footprint`` can be added without exceeding any tier."""
        free = self.headroom(placements)
        return all(footprint.get(t) <= free.get(t) for t in RESIDENT_TIERS)

    def check_admits(
        self,
        footprint: MemoryFootprint,
        placements: Iterable[Placement],
        *,
        model_id: str = "",
    ) -> None:
        """Raise :class:`ResourceExhausted` naming the failing tier.

        Raises:
            ResourceExhausted: When any resident tier would be over budget.
        """
        snapshot = list(placements)
        free = self.headroom(snapshot)
        for tier in RESIDENT_TIERS:
            want = footprint.get(tier)
            if want > free.get(tier):
                who = f" for {model_id!r}" if model_id else ""
                raise ResourceExhausted(
                    f"Cannot admit model{who}",
                    tier=str(tier),
                    requested_bytes=want,
                    available_bytes=free.get(tier),
                    budget_bytes=self.get(tier),
                )

    def is_single_slot_for(self, footprints: Iterable[MemoryFootprint]) -> bool:
        """True when at most one of these models can be resident at once.

        The library assumes this by default: multi-residency is the
        optimisation, never the assumption (``BUILD_BRIEF.md`` section 2).
        """
        cards = list(footprints)
        for i, first in enumerate(cards):
            for second in cards[i + 1 :]:
                if self.fits(first + second):
                    return False
        return True

    def describe(self) -> str:
        """One-line human summary for logs and error messages."""
        parts = [f"{t}={self.get(t) / _GB:.2f}GB" for t in RESIDENT_TIERS]
        return "MemoryBudget(" + ", ".join(parts) + ")"
