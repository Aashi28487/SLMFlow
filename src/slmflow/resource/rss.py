"""Memory measurement that works in-process *and* out-of-process.

``BUILD_BRIEF.md`` section 2: the framework must work when the model runs
in-process (llama-cpp-python) and out-of-process (the Ollama daemon), and
"memory accounting must handle both -- measure RSS of the right process".

So this module exposes:

* :func:`process_rss_bytes` -- RSS of this process (in-process backends).
* :func:`pid_rss_bytes` / :func:`rss_by_name` -- RSS of another process
  (daemon backends).
* :func:`gpu_used_bytes` -- VRAM in use, read from ``nvidia-smi``.

``psutil`` is used when installed; every function has a stdlib fallback so
measurement still works on a bare T0 install. A function that cannot measure
returns ``0`` and logs once, rather than returning a plausible guess.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache

from slmflow.telemetry.trace import get_logger

__all__ = [
    "process_rss_bytes",
    "pid_rss_bytes",
    "rss_by_name",
    "gpu_used_bytes",
    "gpu_process_bytes",
    "measurement_backend",
]

_LOG = get_logger("resource.rss")


@lru_cache(maxsize=1)
def _psutil() -> object | None:
    """Import ``psutil`` once, or return ``None``."""
    try:
        import psutil  # noqa: PLC0415

        return psutil
    except ImportError:
        return None


@lru_cache(maxsize=1)
def measurement_backend() -> str:
    """Name the mechanism RSS measurement will actually use.

    Recorded in results artifacts so a reader knows how the numbers were taken.
    """
    if _psutil() is not None:
        return "psutil"
    if sys.platform == "linux":
        return "/proc/<pid>/statm"
    if sys.platform == "win32":
        return "GetProcessMemoryInfo"
    return "unavailable"


def process_rss_bytes() -> int:
    """Resident set size of *this* process, in bytes. ``0`` if unmeasurable."""
    return pid_rss_bytes(os.getpid())


def pid_rss_bytes(pid: int) -> int:
    """Resident set size of an arbitrary process, in bytes.

    Args:
        pid: Process id to measure.

    Returns:
        Bytes resident, or ``0`` when the process is gone or unmeasurable.
    """
    ps = _psutil()
    if ps is not None:
        try:
            return int(ps.Process(pid).memory_info().rss)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - process died or access denied
            return 0

    if sys.platform == "linux":
        try:
            with open(f"/proc/{pid}/statm", encoding="ascii") as fh:
                fields = fh.read().split()
            pages = int(fields[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return 0

    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415
            from ctypes import wintypes  # noqa: PLC0415

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
            handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
            if not handle:
                return 0
            try:
                counters = _Counters()
                counters.cb = ctypes.sizeof(_Counters)
                ok = psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), ctypes.sizeof(counters)
                )
                return int(counters.WorkingSetSize) if ok else 0
            finally:
                kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001 - ctypes unavailable
            return 0

    return 0


def rss_by_name(name_contains: str) -> tuple[int, list[int]]:
    """Total RSS of every process whose name contains ``name_contains``.

    This is how a daemon backend is measured: the Ollama server holds the
    weights in *its* address space, so measuring our own RSS would report
    almost nothing and silently understate peak memory.

    Args:
        name_contains: Case-insensitive substring, e.g. ``"ollama"``.

    Returns:
        ``(total_bytes, matched_pids)``. ``(0, [])`` when nothing matched or
        process enumeration is unavailable.
    """
    ps = _psutil()
    if ps is None:
        pids = _enumerate_pids_by_name(name_contains)
        return sum(pid_rss_bytes(p) for p in pids), pids

    total = 0
    pids: list[int] = []
    needle = name_contains.lower()
    try:
        for proc in ps.process_iter(["pid", "name"]):  # type: ignore[attr-defined]
            try:
                pname = (proc.info.get("name") or "").lower()
                if needle in pname:
                    total += int(proc.memory_info().rss)
                    pids.append(int(proc.info["pid"]))
            except Exception:  # noqa: BLE001, PERF203 - process vanished mid-iteration
                continue
    except Exception:  # noqa: BLE001 - enumeration refused
        return 0, []
    return total, pids


def _enumerate_pids_by_name(name_contains: str) -> list[int]:
    """Stdlib-only process lookup by name."""
    needle = name_contains.lower()
    if sys.platform == "linux":
        pids: list[int] = []
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry}/comm", encoding="utf-8", errors="replace") as fh:
                        if needle in fh.read().strip().lower():
                            pids.append(int(entry))
                except OSError:  # noqa: PERF203 - process vanished
                    continue
        except OSError:
            return []
        return pids

    if sys.platform == "win32" and shutil.which("tasklist"):
        try:
            out = subprocess.run(  # noqa: S603 - fixed argv
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        found: list[int] = []
        for line in out.splitlines():
            cells = [c.strip('" ') for c in line.split('","')]
            if len(cells) >= 2 and needle in cells[0].lower():
                try:
                    found.append(int(cells[1]))
                except ValueError:  # noqa: PERF203
                    continue
        return found

    return []


def gpu_used_bytes(device_index: int = 0) -> int:
    """VRAM currently in use on one GPU, in bytes.

    Read from ``nvidia-smi`` so no CUDA runtime is imported into this process.
    Returns ``0`` when there is no NVIDIA GPU or the tool is absent -- which is
    also the correct answer for the VRAM tier on a CPU-only machine.
    """
    out = _nvidia_smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if not out:
        return 0
    lines = out.splitlines()
    if device_index >= len(lines):
        return 0
    try:
        return int(float(lines[device_index].strip()) * 1024 * 1024)
    except ValueError:
        return 0


def gpu_process_bytes(pid: int) -> int:
    """VRAM attributed to one process, in bytes.

    More precise than :func:`gpu_used_bytes` when something else is also using
    the GPU (a browser, a compositor). Returns ``0`` when unavailable.
    """
    out = _nvidia_smi(
        [
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                if int(parts[0]) == pid:
                    return int(float(parts[1]) * 1024 * 1024)
            except ValueError:  # noqa: PERF203
                continue
    return 0


def _nvidia_smi(args: list[str], timeout: float = 6.0) -> str:
    """Run ``nvidia-smi`` with ``args``, returning stdout or ``""``."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [exe, *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""
