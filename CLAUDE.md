# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OptimizerV7_G3 is a research-grade GPU optimization system for professional VFX workflows on Windows 11. The system uses a Mixture of Experts (MoE) architecture to automatically optimize three applications: SideFX Houdini 21, Unreal Engine 5.6+, and ComfyUI for a specific hardware configuration (RTX 4090 + AMD Threadripper PRO 7965WX).

**Target Hardware:**
- GPU: NVIDIA GeForce RTX 4090 (24GB VRAM, sm_89 Ada Lovelace)
- CPU: AMD Ryzen Threadripper PRO 7965WX (24 cores/48 threads)
- RAM: 128GB DDR5
- OS: Windows 11 Pro for Workstations
- Driver: 616.92 Studio (exposes CUDA 13.4) — dated 2026-10-01; live value from `python -m v12 truth`
- Display: Apple Pro Display XDR 6016x3384 — its compositor (dwm.exe) held ~2.7-2.9GB VRAM when measured 2026-10-01

## Development Commands

### Environment Setup
```batch
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Running the Optimizer

```batch
# Main launcher (Tier 1 + Tier 2 system)
optimize_rtx4090_vfx.bat

# Next-generation optimizer: ask what you want faster, it measures first
Optimize-The-4090.bat
Optimize-The-4090.bat --request render|shot|compute|profile --reps 5

# Single optimization cycle (no continuous monitoring)
python core/gpu_optimizer_v6.py --config config.yaml --once

# Custom monitoring interval (default: 60 seconds)
python core/gpu_optimizer_v6.py --config config.yaml --interval 120

# Debug mode with verbose expert reasoning
# First, edit config.yaml and set:
#   logging:
#     level: "DEBUG"
#     verbose_experts: true
python core/gpu_optimizer_v6.py --config config.yaml
```

### Measuring (the harness)

```batch
# Next-generation optimizer (see below)
Optimize-The-4090.bat

# Read-only measurements, nothing applied
Measure-This-Machine.bat

# RETIRED - delegates to Optimize-The-4090.bat
The-Method.bat
```

`The-Method.bat` was retired on 2026-11-04 and now only delegates. It used to
gate a candidate against a baseline read from disk — usually measured in an
earlier session, under different resident apps — so drift was charged to the
candidate. That produced the ledger's **ADMIT +12.34% (p=1.9e-05)** for a
thread-count change worth **DENY -0.65%** — nothing at all.
`harness/method.py candidate` and
`harness/lever_run.py` now REFUSE to produce a verdict at all; every lever is
implemented once, in `harness/levers.py`, and measured paired.

That verdict was retracted on 2026-11-04 (`harness/retract.py` marks it
RETRACTED, keeping the original text and adding what replaced it), and the two
baselines it was gated against are **superseded** — numbers intact, but
skipped by lookups and refused by the gate. A baseline can be withdrawn with
`v12 baseline --supersede <id> --reason "..."`; the reason is mandatory and
travels with it, because "this number is gone" and "this number was withdrawn
because it was measured against a stale control" are different facts.

Two more verdicts from that session were retracted the same way:
`aov_beauty_only` (+16.91%, baseline 2h40m earlier) and `aov_no_crypto`
(+11.25%, baseline 2h50m earlier). Re-measured properly, the beauty-only AOV
lever is **+10.85%** (p=1.13e-04) — still real, a third smaller — while
`no_crypto` alone is **-0.09%, DENY**: dropping cryptomatte and nothing else
is worth nothing here. The two AOV reductions are not interchangeable. The
measured levers to trust are the ones in `Optimize-The-4090-Card.md`.

Each was measured more than once, and the table is a HISTORY — every run stays
in it, because what a measurement said at the time is part of the record:

```
beauty_only     10:36  ADMIT  +9.19%   drift +1.45%   contended
beauty_only     23:19  ADMIT  +9.79%   drift -1.82%   verified idle
beauty_only     07:11  ADMIT +10.85%   drift -0.34%   verified idle, current
no_crypto       16:07  DENY   +0.85%   drift +1.59%   contended
no_crypto       22:19  DENY   +0.77%   drift +0.31%   verified idle
no_crypto       07:24  DENY   -0.09%   drift -0.75%   verified idle, current
husk_threads    11:11  DENY   -0.89%   drift +1.96%   contended
husk_threads    05:52  DENY   -0.60%   drift -0.28%   verified idle
husk_threads    07:38  DENY   -0.65%   drift +0.11%   verified idle, current
frames_per_proc 08:33  ADMIT  +47.23%  (no drift field)  old code, no warm-up
frames_per_proc 06:56  ADMIT  +46.95%  drift +2.12%   verified idle, paired
```

The 2026-11-06 runs are the first taken on a SHARED card — Nuke and Houdini
open, idle, holding GPU contexts. The contention is recorded in every one of
them, and the comparison still stands, because the rule is that contention
which leaves the CONTROL steady can only make the effect look SMALLER, never
larger: a false DENY, not a false ADMIT. The drift and spread checks are what
decide, and they are in the table.

`beauty_only` is confirmed across three independent paired runs spanning
+9.19% to +10.85%. Each new measurement supersedes the last; every one stays
in the retraction's history chain, with the reason it stopped being current.

`frames_per_process` is measured twice as well, and the second run is the
authoritative one: **+46.95%, p=2.4e-06, n=3/3, paired, drift +2.12%, zero
contention events**. The 08:33 run reported +47.23% but was taken under the
old code, which had **no warm-up repetition** — the split arm launches three
husk processes, so its first repetition paid the cold-cache cost inside the
measured series, and that run carries no drift field to show it. `Context.warmup`
now defaults to 1, every lever uses `max(ctx.warmup, 1)`, and `--warmup` is on
the CLI. Without it the cold start read as the machine changing and the drift
gate refused the campaign on an idle card with zero contention events.

`husk_threads` was also the one lever with NO quiet gate — it went straight to
measuring without waiting for the card between repetitions. The lever that
produced the phantom had the least protection of any lever. It waits now, and
`tests/test_lever_gates.py` pins every lever to having the gate.

A retraction can be made BEFORE its replacement has been measured; `harness/retract.py
--update` points an existing retraction at a later measurement and records a
corrected reason ALONGSIDE the original rather than over it. Corrections to
the correction chain the same way (`reason_correction_history`): a later
correction must not erase an earlier one.

`harness/apply_retractions.py` moves a retraction's pointer in bulk, from a
mapping file. It also moves the repo's committed copy of the record, because
a retraction edits a record **in place** — the file's name does not change,
which is exactly the case a "copy the new files" sync cannot see. It skips
the sync when the store is a COPY rather than the live one, so a test's
fixture can never overwrite the real evidence.

### Optimize The 4090

`Optimize-The-4090.bat` is the current entry point for optimization. It takes a
*request* rather than a lever, because the card is not the constraint in every
workload:

| Request | Objective | Levers it offers | Why |
|---|---|---|---|
| `compute` | `tflops` | power ceiling 450→600 W | the card IS pinned at the cap (433 W of 450 W) |
| `shot` | `shot_s` | frames per process | ~39 s setup tax paid once per shot, not once per frame |
| `render` | `render_s` | AOV variants | one frame cannot amortize setup; shade fewer pixels instead |

Measured on this box (2026-11-04/05): power ceiling **+10.57%** (p=4.4e-06), frames
per process **+46.95%** on a 3-frame shot. The same power lever does **nothing**
for Karma render, which draws 104 W of 450 W and never reaches the cap.

### The two measurement rules the harness enforces

1. **Baseline and candidate are measured alternately, in one session.** On
   2026-11-03 this repo gated a thread-count sweep against a baseline measured
   48 minutes earlier, under different resident processes: ADMIT +12.34%,
   p=1.9e-05. Re-timed the same way it was **DENY -0.65%** — the Field Notes'
   "move that evaporated". Now both arms run alternately, the gate uses a paired
   t-test, and the baseline's drift across the run is printed and REFUSED if it
   exceeds 5%.
2. **Nothing is compared against a picture of itself.** The ssim guardrail
   scores the candidate's frames against the *baseline's* frames using the same
   reference. The 16-spp data pass renders a completely different picture
   (cross-run SSIM 0.28) yet is perfectly self-consistent at 1.0, so a
   self-comparison would have passed it.

### How contention is detected — process identity, not utilisation

During a Karma render this card is busy **by design** (our own workload
measured at 94% utilisation), so a utilisation threshold cannot tell our render
from somebody else's. An earlier per-rep sampler reported a repetition that
took 148 s — where a clean frame takes 48 s — as CLEAN, because our own render
had made the card busy.

`harness/common.py` asks **who holds the card**, minus our own process tree
(`my_process_tree()`; our renderer is a child of the harness), and polls
**during** each repetition (`ContenderWatcher`) so a rival that arrives and
leaves mid-render is still caught.

Two box-specific facts the detector must know:

- **WSL GPU work is invisible to the Windows process table.** On 2026-11-04 the
  card sat at 98% / 288 W with *no* Windows process accounting for it — the work
  was a python3.12 inside a WSL Ubuntu. The detector queries WSL directly, per
  RUNNING distribution (`wsl --list --running`; `wsl -d <name>` on a stopped
  distro STARTS it, which a detector must never do). V12's
  `guard.wsl_gpu_apps()` does the same, and `v12 preflight` no longer prints
  "CLEAR — nothing is competing" on a pinned card.
- **`dwm.exe` (~2.2 GB) and `csrss.exe` (~385 MB) commit GPU memory
  permanently** and are exempt. Counting them makes every check a false
  positive, and a detector that always says "busy" is worse than none.

Utilisation is still used — but only *before* a repetition starts, when our own
workload is not holding the card. At that moment it catches work that holds no
memory at all. Its threshold is **85%**, calibrated to this desktop (measured
over 113 samples): the desktop alone drives the card to 4-72% at 32-70 W, a
Karma render sits above 90% at 104 W+, and a WSL benchmark at 98% and 288 W.
The threshold used to be 10%, which was *below* this desktop's own idle burst
— so the gate called an idle card busy and waited for a quiet that never came.

### V12 objectives

`render` (render_s), `shot` (shot_s), `diffusion` (frame_ms), `vellum` (solve_s),
`gemm` (tflops). `shot` exists because `render` cannot see the setup tax: it
measures one frame's shading with setup excluded, and setup is what dominates.
The `gemm` objective carries the torch build in its fingerprint, because a
TFLOPS number from a different torch is a different measurement.

`NVML_POWER_LIMIT` is the one NVML target whose prior is READABLE and whose
setter round-trips, so raising the ceiling and undoing it restores the
operator's own number instead of resetting to vendor default.

### Benchmarking

```batch
# Run Houdini synthetic benchmarks
python benchmarks/houdini_synthetic.py

# Run UE5 synthetic benchmarks
python benchmarks/ue5_synthetic.py

# Execute benchmark batch script
run_benchmark.bat

# Post-benchmark analysis
python post_benchmark.py
```

### Diagnostics

```batch
# System diagnostics
diagnose.bat

# Genetic algorithm nightly evolution
nightly_evolution.bat

# First-time installation and run
install_and_run.bat
```

### Dashboard

```batch
# Start web dashboard (FastAPI)
python dashboard/app.py
# Access at: http://localhost:8090
```

## Architecture

### Three-Tier Optimization System

**TIER 1: Immediate Optimization** (Instant)
- Deployed via batch script execution
- Sets CUDA, Houdini, PyTorch/ComfyUI environment variables
- Validates Windows GPU Scheduler
- Impact: +15-25% performance

**TIER 2: Production MoE System** (Continuous)
- 5-expert Mixture of Experts architecture
- Workload detection, context gathering, expert coordination
- Weighted voting based on confidence and impact scores
- Impact: +25-40%, self-improving over time

**TIER 3: Research Components** (Future - Phase 3)
- cuTensor + CUTLASS acceleration
- Specialized TensorRT-LLM model
- Custom OptiX shaders, HDK plugins
- Impact: +40-60% (experimental)

### 5-Expert System

The core orchestrator ([core/gpu_optimizer_v6.py:1-1100](core/gpu_optimizer_v6.py)) implements a Mixture of Experts pattern with 5 specialized experts:

1. **NVIDIA GPU Researcher** (0.70 confidence threshold)
   - Scrapes NVIDIA GitHub repos (CUTLASS, TensorRT-LLM, cuQuantum)
   - Extracts optimization patterns from latest research
   - Weekly update cycle

2. **NVIDIA GPU Specialist** (0.85 confidence threshold)
   - CUDA environment variables (CUDA_CACHE_MAXSIZE, CUBLAS_WORKSPACE_CONFIG)
   - Driver settings and VRAM management
   - Kernel launch parameter tuning

3. **Epic UE5.6 Engineer** (0.80 confidence threshold)
   - Console variables for Lumen, Nanite, VSM, Path Tracer
   - Dynamically modifies Engine.ini
   - Example: `r.Lumen.HardwareRayTracing=1`, `r.Nanite.Streaming.NumInitialRootPages=512`

4. **SideFX Houdini 21 Engineer** (0.80 confidence threshold)
   - Karma XPU render optimization
   - PDG/TOPs GPU scheduling
   - Vellum/Pyro cuTensor acceleration detection
   - MCP houdini-context integration for pattern learning

5. **System Integration Specialist** (0.90 confidence threshold)
   - PCIe Gen4 validation, Resizable BAR checks
   - Windows GPU Scheduler (HwSchMode=2)
   - Thermal monitoring (83°C warning, 90°C critical)
   - Multi-app VRAM budgets (advisory — no GPU partitioning exists on this box)

### Key Data Structures

**WorkflowMode enum:**
- IDLE: No active VFX applications
- SEQUENTIAL: Single application running (18GB VRAM allocation)
- PARALLEL: Multiple applications (cooperative VRAM budgets: 9GB + 9GB + 2GB)
- COMFYUI_ONLY: Only ComfyUI active

**OptimizationAction enum:**
- ENV_VAR: Set environment variables
- REGISTRY: Modify Windows registry
- PROCESS_PRIORITY: Adjust process priorities
- VRAM_ALLOCATION: Per-app VRAM budgets (advisory, not enforced)
- ACCELERATION_LIB: Enable cuTensor/CUTLASS
- QUALITY_SETTINGS: Adjust render quality
- CPU_AFFINITY: Pin a process to a core set (disabled — this CPU has no P/E cores)

**SystemContext dataclass:**
- GPU metrics (utilization, VRAM, temperature, power)
- CPU metrics (usage per core, affinity)
- Running processes (Houdini, UE5, ComfyUI detection)
- VRAM allocation state

**Recommendation dataclass:**
- Expert type (which expert generated this)
- Action type (what to do)
- Confidence score (0.0-1.0)
- Expected impact (performance gain estimate)
- Parameters (action-specific data)

## File Organization

### Core System
- [core/gpu_optimizer_v6.py](core/gpu_optimizer_v6.py) (62KB, ~1100 lines): Main orchestrator with MoE implementation
- [core/genetic_tuner.py](core/genetic_tuner.py) (150+ lines): Evolutionary algorithm for parameter optimization

### Acceleration Modules
- [acceleration/cutlass_integration.py](acceleration/cutlass_integration.py): CUTLASS 3.x integration for RTX 4090 (sm_89 config)
- [acceleration/high_impact_optimizer.py](acceleration/high_impact_optimizer.py): Specialized optimization implementations
- [acceleration/cutensor_vellum_solver.py](acceleration/cutensor_vellum_solver.py): Placeholder for Houdini Vellum cuTensor solver

### Configuration
- [config.yaml](config.yaml) (17.1KB): Hardware profiles, VRAM allocation strategies, expert thresholds, baselines
- [requirements.txt](requirements.txt): Python dependencies (120+ packages)

### Dashboard
- [dashboard/app.py](dashboard/app.py): FastAPI backend (placeholder state, in development)

### Launcher Scripts
- [optimize_rtx4090_vfx.bat](optimize_rtx4090_vfx.bat): Main 3-tier launcher (admin required)
- [diagnose.bat](diagnose.bat): System diagnostics
- [nightly_evolution.bat](nightly_evolution.bat): Genetic algorithm nightly runs
- [run_benchmark.bat](run_benchmark.bat): Benchmark execution

### Data Persistence
- `data/genetic_evolution/`: Evolutionary algorithm state (JSON files)
- `data/optimization_backups/`: Backup optimization states
- `data/nightly_evolution.log`: Nightly evolution logs
- `data/optimizer.log`: Main application logs

## VRAM Management

The system sets **cooperative** per-app VRAM budgets across multiple applications. There is no cross-process GPU partitioning on this box — MIG is datacenter-only, MPS is Linux/QNX-only, and TCC is unreachable on GeForce — so each app is asked (via its own env var / ini) to stay under its budget, and nothing can stop it. "Fencing" here is advisory.

The card has 24564 MiB dedicated, and the Apple Pro Display XDR compositor (dwm.exe) held ~2.7-2.9GB of it when measured 2026-10-01, so app budgets must sum under ~21GB to leave real headroom.

**Sequential Mode** (single app active):
- Active app: 18GB
- Reserve buffer: 2GB
- System + XDR compositor: ~3GB

**Parallel Mode** (Houdini + UE5 + ComfyUI):
- Houdini: 9GB
- UE5: 9GB
- ComfyUI: 2GB
- System + XDR compositor: ~3GB (total: ~23GB of 24GB, ~1GB headroom)

**Fencing Parameters** ([config.yaml](config.yaml)) — advisory thresholds; nothing reads them
today. VRAM fencing is unimplemented (`# TODO: Implement VRAM fencing`,
[core/gpu_optimizer_v6.py:1376](core/gpu_optimizer_v6.py)), and on this hardware no fence
could cap a running process anyway:
- Soft limit: 90% (advisory threshold; no warning is emitted)
- Hard limit: 95% (advisory threshold; nothing enforces it)
- Monitoring interval: 100ms (nothing polls at it)

Mode detection is automatic via process monitoring (psutil).

## Environment Variables Set by Tier 1

**CUDA:**
```batch
CUDA_CACHE_MAXSIZE=4294967296  # 4GB shader cache
CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_MODULE_LOADING=LAZY
```

**Houdini:**
```batch
HOUDINI_OCL_MEMSIZE_MB=18432  # 18GB OpenCL memory: the sequential budget. The launcher sets it
                              # unconditionally, so in parallel mode it exceeds the 9GB Houdini budget.
HOUDINI_PDG_MAX_CONCURRENT_COOKS=24  # = physical cores; the CPU has 48 threads
```

**PyTorch/ComfyUI:**
```batch
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.9
FORCE_MEM_EFFICIENT_ATTN=1
```

These are set automatically by [optimize_rtx4090_vfx.bat](optimize_rtx4090_vfx.bat) before launching the orchestrator.

## Genetic Algorithm

[core/genetic_tuner.py](core/genetic_tuner.py) implements evolutionary optimization:

**Parameters:**
- Population size: 5 genomes
- Mutation rate: 0.2 (20%)
- Elite size: 1 (best genome preserved)
- Fitness function: FPS/VRAM ratio

**Gene Types:**
- Integer: Discrete values (e.g., sample counts)
- Float: Continuous values (e.g., quality multipliers)
- Boolean: On/off flags

**State Persistence:**
- Saves to `data/genetic_evolution/*.json`
- Nightly runs via [nightly_evolution.bat](nightly_evolution.bat)

## Integrations

### vfx-deterministic-llm API
- Location: `C:\Users\User\home\vfx-deterministic-llm\`
- Endpoint: `http://localhost:8000/api/optimize`
- Purpose: TensorRT-LLM-based decision engine
- Timeout: 5 seconds
- Status: Phase 2 (planned integration)

### MCP houdini-context Server
- Tools: `recall_similar_patterns`, `remember_hdk_pattern`
- Workflow: Query MCP for similar scenes → recall historical optimizations → incorporate into expert recommendations
- Status: Phase 2 (planned integration)

### ComfyUI VRAM Coordination
- Auto-detection: Monitors `python.exe` with `ComfyUI\main.py` in command line
- VRAM limit: 2GB when other apps active
- Status: Implemented in orchestrator

## Regression Detection

**Statistical Method:**
- Welch's t-test (p < 0.05)
- Effect size threshold: 10%
- Frequency: Weekly (scheduled via cron)

**Baselines** (defined in [config.yaml](config.yaml)):
- Karma XPU render: 45s target
- PDG/TOPs cook: 120s target
- Vellum solve: 2.5s/frame target
- Lumen GI build: 180s target
- Nanite streaming: 0 hitches/min target

Automatic rollback if regression detected.

## Hardware-Specific Optimizations

### RTX 4090 (sm_89) Features
- 16384 CUDA cores
- 512 4th-gen Tensor Cores
- 72MB L2 Cache
- 100KB shared memory per SM
- FP8 Tensor Cores (2x FP16 throughput)
- Shader execution reordering

**Not features of this GPU, despite appearing in older revisions of this file:**
- *Thread block clusters* — require compute capability 9.0+ (Hopper). Ada is sm_89
  and does not support them. Anything that assumes clusters work here will either
  fail to compile or silently fall back.
- *FP4 / NVFP4* — sm_100 family only. Hard-blocked on sm_89.

These are configured in [acceleration/cutlass_integration.py](acceleration/cutlass_integration.py) via the `SM89Config` dataclass.

### AMD Threadripper PRO 7965WX
- 24 cores / 48 logical processors, 4 CCDs
- **No P/E cores** — all 24 are identical Zen 4 cores. Older revisions of this file split them into "performance 0-20 / background 21-23"; applied, that confines Houdini/UE to 21 of 48 LPs.
- CPU pinning is disabled (`enable_pinning: false`) in [config.yaml](config.yaml) under `hardware.cpu.affinity` and in `v9/config.yaml` under `system`

## Development Phase Status

**Phase 1 (Complete):**
- Core orchestrator with 5-expert MoE
- Tier 1 environment variable optimizations
- Configuration system
- Main launcher script

**Phase 2 (In Progress):**
- Telemetry monitor (pynvml + WebSocket)
- Web dashboard (FastAPI + React) - currently placeholder
- Synthetic benchmarks - empty placeholders
- Regression detection system
- cuTensor Python SOP prototype

**Phase 3 (Future):**
- Custom CUTLASS kernels
- Fine-tuned TensorRT-LLM model
- HDK plugins (cuTensor Vellum solver)
- Custom OptiX shaders

## Important Notes for Development

1. **Admin Privileges Required**: The launcher script modifies Windows registry (GPU Scheduler) and sets system-level environment variables. Always run as Administrator.

2. **Hardware-Specific**: This system is optimized for RTX 4090 (sm_89) and Threadripper PRO 7965WX. Don't generalize optimizations without hardware validation.

3. **Configuration-Driven**: All thresholds, allocations, and expert parameters are in [config.yaml](config.yaml). Always check config before modifying code.

4. **Expert Confidence Thresholds**: Each expert has a minimum confidence threshold. Recommendations below this threshold are discarded. Adjust in config to control conservativeness.

5. **VRAM Budgets are Cooperative**: VRAM management reduces OOM crashes in parallel workflows, but nothing partitions the GPU (no MIG/MPS/TCC here), so budgets are requests, not fences. App budgets are 9GB + 9GB + 2GB, plus ~3GB for the system and XDR compositor — ~23GB of the 24GB. Don't exceed these limits.

6. **Logging is Verbose**: Set `logging.level: "DEBUG"` and `verbose_experts: true` in config for detailed expert reasoning. Check `data/optimizer.log` for diagnostics.

7. **Benchmark Suite Incomplete**: The `benchmarks/` directory contains placeholders. Synthetic benchmarks for Houdini and UE5 are planned but not implemented.

8. **Dashboard is Placeholder**: The web UI at http://localhost:8090 is minimal. Full telemetry WebSocket and React frontend are in development.

9. **Windows-Specific**: Uses pywin32 for registry access, Windows GPU Scheduler validation, and process management. Not portable to Linux/macOS.

10. **Genetic Algorithm Runs Nightly**: The evolutionary tuner runs via [nightly_evolution.bat](nightly_evolution.bat) to optimize parameters over time. State is preserved in `data/genetic_evolution/`.
