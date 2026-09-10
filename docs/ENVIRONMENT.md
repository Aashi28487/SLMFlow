# Environment

> **NOT YET PROBED.** Phase 0's Definition of Done requires this file to
> contain **real probed values** obtained by running code, not assumptions.
> Nothing in this repository has been executed, so the table below is empty by
> design rather than filled with plausible guesses (Ground Rule 1).
>
> Generate it on the target machine:
>
> ```
> slmflow probe > docs/ENVIRONMENT.md
> ```
>
> Then open the file and add the exact CPU and GPU model numbers if the probe
> reported them vaguely — `PROCESSOR_IDENTIFIER` on Windows is often generic.

## Expected machine (to be confirmed by the probe, not trusted)

| Property | Stated | Probed |
|---|---|---|
| CPU | Intel i5, HX-series (high-power mobile) | not yet probed |
| Logical cores | unknown | not yet probed |
| System RAM | 16 GB | not yet probed |
| GPU | discrete, ~6 GB VRAM | not yet probed |
| CUDA available | expected yes | not yet probed |
| Disk | 512 GB SSD, shared with `data/` | not yet probed |
| OS | Windows | not yet probed |
| Python | 3.10-3.13 recommended (3.12 safest) | not yet probed |
| Device tier | expected T1 | not yet probed |
| RSS measurement backend | psutil if installed | not yet probed |

## Why the probe matters more than the spec sheet

Three values change the experiment and cannot be taken from a spec sheet:

1. **Usable VRAM, not total VRAM.** The desktop compositor and any browser
   already hold some. The budget is derived from what is *free*, with 25 %
   headroom left for the OS.
2. **How VRAM was read.** If `nvidia-smi` is not on PATH the probe falls back to
   the Windows `Win32_VideoController.AdapterRAM` field, which is 32-bit and
   **saturates at 4 GB**. A 6 GB card then reports as 4 GB and every memory
   number is quietly wrong. `slmflow doctor` warns about this explicitly, and
   the probe records which source it used.
3. **Available RAM at run time.** 16 GB total is not 16 GB usable. The derived
   budget uses available, not total.

## Reproducibility record

Ground Rule 11 requires these in every results artifact. The harness collects
them automatically via `benchmarks.harness.environment_record`:

- CPU model, logical cores, total RAM
- GPU model, total VRAM, CUDA availability
- device tier, platform string, Python version
- RSS measurement mechanism (`psutil` / `/proc` / `GetProcessMemoryInfo`)
- pinned model digests for every registered model
- any probe warnings, carried through rather than dropped
