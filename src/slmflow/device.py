"""Device probing. Produces a measured :class:`DeviceCard` for *this* machine.

Half of the C2 contribution is a device card that was measured rather than
assumed, so nothing here guesses. Where a value cannot be read, the field is
left at zero and :func:`probe` records the reason in ``notes`` rather than
substituting a plausible number (Ground Rule 1).

Everything uses stdlib only. ``psutil`` and ``nvidia-smi`` are consulted when
present but are never required; the pure-stdlib path must work on T0.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from slmflow.types import DeviceCard, DeviceTier, MemoryTier

__all__ = ["probe", "ProbeReport", "classify_tier", "budget_from_env"]

_GB = 1_000_000_000


@dataclass(slots=True)
class ProbeReport:
    """A device card plus the provenance of every field that was read."""

    card: DeviceCard
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_markdown(self) -> str:
        """Render the probe as a table for ``docs/ENVIRONMENT.md``."""
        c = self.card
        rows = [
            ("Device tier", str(c.tier)),
            ("CPU", c.cpu_model or "not detected"),
            ("Logical cores", str(c.logical_cores)),
            ("Total RAM", f"{c.total_ram_bytes / _GB:.2f} GB"),
            ("Available RAM", f"{c.available_ram_bytes / _GB:.2f} GB"),
            ("GPU", c.gpu_model or "none detected"),
            ("Total VRAM", f"{c.total_vram_bytes / _GB:.2f} GB" if c.has_gpu else "0 (no GPU)"),
            ("CUDA available", "yes" if c.cuda_available else "no"),
            ("Free disk", f"{c.free_disk_bytes / _GB:.2f} GB"),
            ("Platform", c.platform),
            ("Python", c.python_version),
            ("Probed at", c.probed_at),
        ]
        lines = ["| Property | Measured value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rows]
        if self.sources:
            lines += ["", "| Field | Read from |", "|---|---|"]
            lines += [f"| {k} | `{v}` |" for k, v in sorted(self.sources.items())]
        if self.warnings:
            lines += ["", "**Warnings**", ""]
            lines += [f"- {w}" for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# RAM
# --------------------------------------------------------------------------- #
def _ram_bytes() -> tuple[int, int, str]:
    """Return ``(total, available, source)`` in bytes."""
    try:
        import psutil  # noqa: PLC0415 - optional, probed lazily

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available), "psutil.virtual_memory"
    except Exception:  # noqa: BLE001 - psutil absent or refused; fall through
        pass

    if hasattr(os, "sysconf"):
        try:
            names = os.sysconf_names  # type: ignore[attr-defined]
            page = os.sysconf("SC_PAGE_SIZE")
            total = os.sysconf("SC_PHYS_PAGES") * page
            avail_key = "SC_AVPHYS_PAGES"
            avail = os.sysconf(avail_key) * page if avail_key in names else total // 2
            return int(total), int(avail), "os.sysconf"
        except (OSError, ValueError, KeyError):
            pass

    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys), "GlobalMemoryStatusEx"
        except Exception:  # noqa: BLE001 - ctypes unavailable
            pass

    return 0, 0, "unavailable"


# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #
def _cpu_model() -> tuple[str, str]:
    """Return ``(model_string, source)``."""
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip(), "/proc/cpuinfo"
        except OSError:
            pass
    if sys.platform == "win32":
        env = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if env:
            return env, "env:PROCESSOR_IDENTIFIER"
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out, "sysctl machdep.cpu.brand_string"
    return platform.processor() or platform.machine(), "platform.processor"


def _run(cmd: list[str], timeout: float = 6.0) -> str:
    """Run a probe command, returning stripped stdout or ``""`` on any failure."""
    if shutil.which(cmd[0]) is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def _gpu() -> tuple[int, str, bool, str]:
    """Return ``(total_vram_bytes, gpu_model, cuda_available, source)``.

    Tries ``nvidia-smi`` first because it reports real VRAM without importing a
    CUDA runtime into this process. Never raises.
    """
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if out:
        first = out.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 2:
            try:
                mib = float(parts[1])
                return int(mib * 1024 * 1024), parts[0], True, "nvidia-smi"
            except ValueError:
                pass

    # Windows fallback: WMI via PowerShell. AdapterRAM is a 32-bit field and
    # saturates at 4 GB, so flag it rather than trust it.
    if sys.platform == "win32":
        ps = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController | "
                "Select-Object -First 1 Name,AdapterRAM | ConvertTo-Csv -NoTypeInformation)",
            ]
        )
        lines = [ln for ln in ps.splitlines() if ln.strip()]
        if len(lines) >= 2:
            cells = [c.strip().strip('"') for c in lines[1].split(",")]
            if len(cells) >= 2:
                name = cells[0]
                try:
                    return int(cells[1]), name, False, "Win32_VideoController (AdapterRAM)"
                except ValueError:
                    return 0, name, False, "Win32_VideoController (name only)"
    return 0, "", False, "unavailable"


# --------------------------------------------------------------------------- #
# Tier classification
# --------------------------------------------------------------------------- #
def classify_tier(total_ram_bytes: int, total_vram_bytes: int, is_arm_mobile: bool) -> DeviceTier:
    """Classify a machine into a tier from ``BUILD_BRIEF.md`` section 2.

    Args:
        total_ram_bytes: Physical RAM.
        total_vram_bytes: Dedicated VRAM, 0 when absent.
        is_arm_mobile: True on Android/Termux or an ARM SBC.

    Returns:
        The tier whose design rules apply.
    """
    if is_arm_mobile or total_ram_bytes < 6 * _GB:
        return DeviceTier.T0_PHONE
    if total_vram_bytes >= 12 * _GB:
        return DeviceTier.T2_WORKSTATION
    return DeviceTier.T1_LAPTOP


def _is_arm_mobile() -> bool:
    """Heuristic detection of Android/Termux or an ARM single-board computer."""
    if "ANDROID_ROOT" in os.environ or "TERMUX_VERSION" in os.environ:
        return True
    machine = platform.machine().lower()
    return machine.startswith(("aarch64", "armv")) and sys.platform.startswith("linux")


def probe(disk_path: str | None = None) -> ProbeReport:
    """Measure this machine and return a :class:`ProbeReport`.

    Args:
        disk_path: Filesystem to report free space for. Defaults to the
            current working directory, which is where model weights land.

    Returns:
        A report whose ``card`` is safe to persist into a results artifact.

    Example:
        >>> report = probe()
        >>> report.card.logical_cores > 0
        True
    """
    warnings: list[str] = []
    sources: dict[str, str] = {}

    total_ram, avail_ram, ram_src = _ram_bytes()
    sources["ram"] = ram_src
    if total_ram == 0:
        warnings.append("Could not read physical RAM; budgets must be set explicitly.")

    cpu, cpu_src = _cpu_model()
    sources["cpu_model"] = cpu_src

    vram, gpu_name, cuda, gpu_src = _gpu()
    sources["gpu"] = gpu_src
    if gpu_src.startswith("Win32_VideoController"):
        warnings.append(
            "VRAM came from Win32_VideoController.AdapterRAM, a 32-bit field that saturates "
            "at 4 GB and is unreliable. Install the NVIDIA driver tools so nvidia-smi is on "
            "PATH, or set SLMFLOW_VRAM_BUDGET explicitly."
        )
    if vram == 0:
        warnings.append("No GPU VRAM detected; the VRAM tier budget will be 0.")

    path = disk_path or os.getcwd()
    try:
        free_disk = shutil.disk_usage(path).free
        sources["disk"] = f"shutil.disk_usage({path})"
    except OSError:
        free_disk = 0
        sources["disk"] = "unavailable"
        warnings.append(f"Could not stat {path} for free disk space.")

    card = DeviceCard(
        tier=classify_tier(total_ram, vram, _is_arm_mobile()),
        cpu_model=cpu,
        logical_cores=os.cpu_count() or 1,
        total_ram_bytes=total_ram,
        available_ram_bytes=avail_ram,
        total_vram_bytes=vram,
        gpu_model=gpu_name,
        cuda_available=cuda,
        free_disk_bytes=free_disk,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=platform.python_version(),
        probed_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    return ProbeReport(card=card, sources=sources, warnings=warnings)


def budget_from_env(card: DeviceCard, headroom: float = 0.75) -> dict[MemoryTier, int]:
    """Per-tier budget, with environment overrides.

    ``SLMFLOW_VRAM_BUDGET`` and ``SLMFLOW_RAM_BUDGET`` accept a byte count or a
    ``"5GB"`` style suffix and take precedence over the probed value. Use them
    when a probe is unreliable, or to force a small budget for an experiment.

    Args:
        card: The probed device card.
        headroom: Fraction of each tier to claim when no override is set.

    Returns:
        Mapping of tier to byte budget.
    """
    budget = card.default_budget_bytes(headroom)
    for tier, var in (
        (MemoryTier.VRAM, "SLMFLOW_VRAM_BUDGET"),
        (MemoryTier.RAM, "SLMFLOW_RAM_BUDGET"),
    ):
        raw = os.environ.get(var)
        if raw:
            budget[tier] = _parse_bytes(raw, var)
    return budget


def _parse_bytes(raw: str, var: str) -> int:
    """Parse ``"5GB"`` / ``"512MB"`` / a plain byte count."""
    text = raw.strip().upper().replace("I", "")
    multipliers = {"KB": 1_000, "MB": 1_000_000, "GB": 1_000_000_000, "TB": 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * mult)
            except ValueError as exc:
                raise ValueError(f"{var}={raw!r} is not a valid size.") from exc
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(
            f"{var}={raw!r} is not a valid size. Use bytes or a suffix like '5GB'."
        ) from exc
