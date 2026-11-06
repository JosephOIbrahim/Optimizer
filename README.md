# Optimizer — a measurement harness that tells you the truth about your GPU

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A harness for measuring GPU levers on your own card, built around the
failure mode that makes every other harness wrong.**

This repository exists because of a specific false result, and the false
result is the reason for most of what is in it.

On 2026-11-03 this harness reported **ADMIT +12.34%** for a thread-count
change that was worth **nothing**. The cause was not a statistical bug. The
candidate was measured at 19:29 and gated against a baseline measured at
18:41 — 48 minutes earlier, under a different set of resident processes.
Welch's t-test on two *independent* series cannot distinguish drift from
effect; that is what "independent" means. The two numbers described two
different machine states, and the difference was charged to the candidate.

Three verdicts were retracted because of it. All three are still in the
records, marked, with the original claims visible.

**Nothing here is the lever. The gates are.** A lever that wins by 0.5% is
indistinguishable from a machine that drifted by 0.5%, and the only honest
way to tell them apart is to measure both arms in the same session and then
check the control.

---

## The gates

Every one of these exists because a campaign got past the gate that should
have stopped it.

| Gate | What it refuses | The failure it was written for |
|---|---|---|
| **Paired arms** | a comparison whose two arms were measured in different sessions | the +12.34% phantom above |
| **Drift** | a control whose *first* repetition differs from its *last* | a control that changed state mid-run |
| **Control spread** | a control whose *largest* deviation from its own median is too large | a busy window in the **middle** of a run — end-to-end drift reads +0.00% and passes a control that moved enormously |
| **Warm-up** | a series whose first repetition paid for a cold cache | a cold page cache, shader cache and BVH landing *inside* the measured series, read as drift |
| **Quiet budget** | a campaign that waits for the card twice | the timeout was applied once before the lever and again per repetition |
| **Contention** | — (it discloses) | a rival on the card, detected by **process identity**, not by utilisation |
| **SSIM guardrail** | a candidate that changed the picture | the bar was the reference frame scored against *itself*, so the baseline side carried a free 1.0 nobody rendered |

Contention is the one that does not refuse, and the reasoning is worth
reading: a rival that hits the **control** moves it, and the drift and
spread checks catch it. A rival that hits the **candidate** makes the
candidate read slow, so the effect is *understated* — a false DENY, never a
false ADMIT. Contention that leaves the control steady can only move a
verdict towards refusing it.

## The retraction chain

A verdict that turns out to be wrong is not deleted. It is **marked**, and
the measurement that replaced it is recorded beside it:

```
RETRACTED — was: ADMIT  (+16.91%)  +16.91% improvement, p=2.183e-10, n=5/5
```

The original text stays. The replacement pointer moves, and every
measurement it displaced goes into a history chain with the reason it
stopped being current. A corrected *reason* is recorded **alongside** the
original rather than over it, because the original reason was true when it
was written and a reader who sees only the newest text cannot tell that.

So a reader who greps for an old admission still finds it, marked, with the
correction next to it.

## What it measures, on this box

| lever | workload | effect |
|---|---|---|
| frames per process 1→3 | Karma render, a shot | **+46.95%**, p=2.4e-06 |
| power ceiling 450→600 W | fp16 GEMM | **+10.57%**, p=4.4e-06 |
| AOV set → beauty only | Karma render, a frame | **+10.85%**, p=1.13e-04 |
| AOV set → no crypto | Karma render, a frame | -0.09%, DENY |
| thread count default vs `-j 48` | Karma render, a frame | -0.65%, DENY |

Records carry the raw paired series, so the effect, the p-value and the
drift all re-derive from the file alone. A record that keeps only summary
statistics cannot be *checked* — which is the position this repo was in
before the retraction.

## Re-deriving one measurement yourself

The table above is a claim until you check it. Here is one measurement, end
to end, so you can see that the published figure is arithmetic rather than
assertion — and so you can run it yourself on your own card.

The record is
[`measurements/optimize_20261106_071113.json`](measurements/optimize_20261106_071113.json):
the beauty-only AOV reduction, five repetitions, both arms alternating in
one session.

**The raw series** — seconds per frame, and the difference between the arms
in each repetition:

```json
"baseline_series":  [47.27, 48.30, 48.31, 48.07, 47.11],
"candidate_series": [43.32, 43.68, 43.03, 42.20, 40.90],
"paired_diffs":     [-3.95, -4.62, -5.28, -5.87, -6.21],
```

**The effect.** Render time is lower-is-better, so a positive effect means
the candidate was faster:

```
effect = (mean(baseline) - mean(candidate)) / mean(baseline)
       = (47.812 - 42.626) / 47.812
       = 0.10846649376725509
       = +10.85%
```

**The p-value.** A one-sided paired t-test on the *differences*, not on the
two series. That is the whole point of the pairing: the arms were measured
alternately in one session, so the machine's drift is common to both and
cancels in the difference. Welch's test on two independent series cannot do
that — it attributes the drift to the candidate, which is how the +12.34%
phantom above happened.

```
mean(diff) = -5.186
sd(diff)   = sqrt( sum((d - mean)^2) / (n - 1) ) = 0.918330
se         = sd / sqrt(5)                        = 0.410690
t          = mean / se                          = -12.627540
                                                -> flipped, lower is better: 12.627540
df         = n - 1                               = 4
p          = one-sided t(12.627540, 4)          = 0.00011321489170426711
```

**The gates, from the same series.** The control is the baseline arm:

```
drift          = (last - first) / |first|
               = (47.11 - 47.27) / 47.27        = -0.34%
control spread = max|d - median| / median
               = 0.96 / 48.07                   =  2.00%
```

Both inside the 5% limit, so the comparison stands. The spread is the check
that catches a busy window in the **middle** of a run: end-to-end drift
reads `+0.00%` when the first and last repetitions happen to match, and
passes a control that moved enormously.

**Do it yourself:**

```python
import json, math, pathlib

d = json.loads(pathlib.Path(
    "measurements/optimize_20261106_071113.json").read_text(encoding="utf-8"))
v = d["results"]["aov_variant"]
B, C, D = v["baseline_series"], v["candidate_series"], v["paired_diffs"]

b, c = sum(B) / len(B), sum(C) / len(C)
print("effect", (b - c) / b)                  # 0.10846649376725509

n = len(D)
m = sum(D) / n
sd = math.sqrt(sum((x - m) ** 2 for x in D) / (n - 1))
t = m / (sd / math.sqrt(n))
print("t", -t, "df", n - 1)                    # 12.627539615387068 df 4
```

The p-value needs the t distribution, which the harness implements
dependency-free in `OptimizerV12/v12/objective.py` — `paired_t_test` for
paired series, `welch_t_test` for unpaired. Import it rather than
reimplementing it, and you will get the same number the record stores.

## Running it

```
Optimize-The-4090.bat --request render|shot|compute|profile
                       [--reps N] [--warmup N] [--levers a,b]
```

`--request` names the *workload*, not the lever, because the card is not the
constraint in every workload. The launcher stages the measurement record
and commits it with the campaign, so what was measured and the evidence for
it land in the same commit.

**Start here:**

- [Re-deriving one measurement yourself](#re-deriving-one-measurement-yourself)
  — the whole method in one worked example, from the raw series to the
  published figure
- [`Optimize-The-4090-Card.md`](Optimize-The-4090-Card.md) — the levers, the
  gates, and what each verdict means
- [`Harness-Operator-Card.md`](Harness-Operator-Card.md) — running it, and
  what to do when it breaks
- [`measurements/README.md`](measurements/README.md) — the record format, and
  how the retraction chain is stored
- [`CLAUDE.md`](CLAUDE.md) — the design decisions, with the reasoning

Every published figure in the cards and in the source is machine-checked
against the records: `tests/test_published_figures.py` and
`tests/test_source_verdicts.py` fail if a document publishes a number no
record holds, publishes a superseded number as current, or leaves a figure
undeclared.

---

# OptimizerV6 - RTX 4090 VFX Workstation Optimizer

> **This is the earlier system, kept for reference.** The measurement
> harness above is what this repository is now; the material below documents
> the V6 design it grew out of and still shares a machine with.

**Research-grade GPU optimization system for professional VFX workflows**

Integrates NVIDIA's latest research: **Jet-Nemotron** (multi-agent reasoning), **TensorRT-LLM** (AI-driven decisions), **cuTensor** (tensor network acceleration), **CUTLASS 3.x** (optimized CUDA kernels).

---

## 🎯 System Overview

**Hardware Target:**
- **GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM, Compute Capability 8.9)
- **CPU:** AMD Threadripper PRO 7965WX (24-core, Zen 4)
- **RAM:** 128GB DDR5
- **OS:** Windows 11 Pro for Workstations

**Applications Optimized:**
- **SideFX Houdini 21:** Karma XPU, PDG/TOPs, Vellum, Pyro simulations
- **Unreal Engine 5.6+:** Lumen GI, Nanite, Virtual Shadow Maps, Path Tracer
- **ComfyUI:** Stable Diffusion and other diffusion models

**Performance Targets:**
- Karma XPU render: **-20% frame time**
- PDG/TOPs cook: **-15% cook time**
- Vellum/Pyro solve: **-25% solve time** (with cuTensor)
- UE5 Lumen GI: **-25% compilation time**
- UE5 Nanite: **-30% streaming hitches**
- **Zero VRAM-related crashes** in parallel workflows

---

## 🏗️ Architecture

### Three-Tier System

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: IMMEDIATE OPTIMIZATION                              │
│ ├─ CUDA Environment Variables (cache, memory)              │
│ ├─ Houdini Environment Variables (VRAM, GPU device)         │
│ ├─ PyTorch/ComfyUI Memory Management                       │
│ └─ Windows GPU Scheduler Validation                        │
│ ⚡ Deployment: Single .bat execution                        │
│ 📊 Impact: +15-25% performance, instant                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TIER 2: PRODUCTION MoE SYSTEM                               │
│                                                             │
│ ┌─────────────────────────────────────────────┐            │
│ │ ORCHESTRATOR (Jet-Nemotron Pattern)         │            │
│ │ ├─ Workload Detection (psutil)              │            │
│ │ ├─ Context Gathering (pynvml GPU metrics)   │            │
│ │ ├─ Expert Coordination (parallel threads)   │            │
│ │ ├─ Weighted Voting (confidence × impact)    │            │
│ │ └─ Optimization Application & Validation    │            │
│ └─────────────────────────────────────────────┘            │
│          │                                                  │
│          ├──> Expert 1: NVIDIA GPU Researcher              │
│          │    └─ Scrapes latest research from NVIDIA       │
│          │                                                  │
│          ├──> Expert 2: NVIDIA GPU Specialist              │
│          │    └─ CUDA env vars, driver optimizations       │
│          │                                                  │
│          ├──> Expert 3: Epic UE5.6 Engineer                │
│          │    └─ Console variables (r.Lumen.*, r.Nanite.*)  │
│          │                                                  │
│          ├──> Expert 4: SideFX Houdini 21 Engineer         │
│          │    └─ Karma XPU, PDG, cuTensor acceleration     │
│          │                                                  │
│          └──> Expert 5: System Integration Specialist      │
│               └─ VRAM fencing, thermal monitoring          │
│                                                             │
│ 🔗 Integrations:                                            │
│ ├─ vfx-deterministic-llm API (TensorRT-LLM decisions)      │
│ ├─ MCP houdini-context (pattern learning)                  │
│ └─ Regression Detection (SQLite baselines)                 │
│                                                             │
│ ⚡ Deployment: Background service                           │
│ 📊 Impact: +25-40%, self-improving over time               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TIER 3: RESEARCH COMPONENTS (Phase 3)                       │
│ 🧬 cuTensor + CUTLASS Acceleration Stack                   │
│ 🤖 Specialized TensorRT-LLM Model (fine-tuned)             │
│ 🔬 Custom OptiX Shaders, HDK Plugins                       │
│ ⚡ Deployment: Opt-in after data collection                 │
│ 📊 Impact: +40-60% (experimental)                          │
└─────────────────────────────────────────────────────────────┘
```

### 5-Expert Mixture of Experts (MoE)

Each expert analyzes the workload independently and returns recommendations with confidence scores. The orchestrator aggregates recommendations using weighted voting.

| Expert | Domain | Confidence Threshold |
|--------|--------|---------------------|
| **NVIDIA Researcher** | Latest research patterns (CUTLASS, TensorRT-LLM) | 0.70 |
| **GPU Specialist** | CUDA env vars, driver settings, VRAM management | 0.85 |
| **UE5 Engineer** | Console variables (Lumen, Nanite, VSM, Path Tracer) | 0.80 |
| **Houdini Engineer** | Karma XPU, PDG/TOPs, Vellum, Pyro + MCP integration | 0.80 |
| **System Integrator** | VRAM fencing, PCIe validation, thermal monitoring | 0.90 |

---

## 📦 Installation

### Prerequisites

- **Python 3.10+** (tested with Python 3.13.7)
- **CUDA 12.6+** (for GPU features)
- **NVIDIA Driver 581.80+** (Game Ready or Studio Driver)
- **Git** (for cloning repositories)
- **Administrator privileges** (for registry tweaks and Windows GPU Scheduler)

### Quick Start

1. **Extract OptimizerV6**:
   ```
   Already at: C:\Users\User\OneDrive\Desktop\Dads_Toolbox_For_Repairs\OptimizerV6\
   ```

2. **Install Python Dependencies** (automatic on first run):
   ```bat
   cd OptimizerV6
   optimize_rtx4090_vfx.bat
   ```

   The launcher will:
   - Create Python virtual environment
   - Install all dependencies from `requirements.txt`
   - Configure CUDA/Houdini/ComfyUI environment variables
   - Launch orchestrator and dashboard

3. **Verify Installation**:
   - Dashboard should open at http://localhost:8090
   - Orchestrator logs appear in terminal
   - Check `data/optimizer.log` for detailed logs

### Manual Installation

If you prefer manual setup:

```bat
cd OptimizerV6

REM Create virtual environment
python -m venv venv

REM Activate
venv\Scripts\activate.bat

REM Install dependencies
pip install -r requirements.txt

REM Run orchestrator
python core/gpu_optimizer_v6.py --config config.yaml
```

---

## 🚀 Usage

### Basic Usage

**Run OptimizerV6:**
```bat
optimize_rtx4090_vfx.bat
```

This starts:
- **Tier 1**: Immediate env var optimizations (instant)
- **Tier 2**: Orchestrator with 5-expert MoE system (continuous monitoring)
- **Dashboard**: Web UI at http://localhost:8090

**Single Optimization Cycle** (no continuous monitoring):
```bat
python core/gpu_optimizer_v6.py --config config.yaml --once
```

### Advanced Usage

**Custom monitoring interval** (default: 60 seconds):
```bat
python core/gpu_optimizer_v6.py --config config.yaml --interval 120
```

**Debug mode** (verbose expert reasoning):
Edit `config.yaml`:
```yaml
logging:
  level: "DEBUG"
  verbose_experts: true
```

### Dashboard

The web dashboard (http://localhost:8090) provides:
- **Real-time GPU telemetry** (utilization, VRAM, temp, power)
- **Per-process VRAM breakdown** (Houdini, UE5, ComfyUI)
- **Active optimizations log** (what's being applied)
- **Expert consensus view** (what each expert recommends)
- **Performance baseline comparison** (regression detection)

---

## ⚙️ Configuration

All settings are in `config.yaml`:

### VRAM Allocation

**Sequential Workflow** (default):
```yaml
vram_allocation:
  sequential:
    active_app_gb: 18  # 18GB to active app
    reserve_buffer_gb: 4
```

**Parallel Workflow** (both Houdini + UE5 running):
```yaml
vram_allocation:
  parallel:
    houdini_gb: 10
    ue5_gb: 10
    comfyui_gb: 2
```

The orchestrator automatically detects mode based on running processes.

### Expert Tuning

**Enable/disable experts**:
```yaml
experts:
  nvidia_researcher:
    enabled: true  # Set false to disable
    min_confidence: 0.70
```

**Adjust confidence thresholds**:
```yaml
experts:
  gpu_specialist:
    min_confidence: 0.90  # Increase for more conservative recommendations
```

### Baseline Targets

Define expected performance for regression detection:
```yaml
baselines:
  houdini:
    karma_xpu_render:
      target_seconds_per_frame: 45.0

  ue5:
    lumen_gi_build:
      target_seconds: 180.0
```

---

## 🔬 Integrations

### 1. vfx-deterministic-llm API

Located at: `C:\Users\User\home\vfx-deterministic-llm\`

**Configuration** (`config.yaml`):
```yaml
tensorrt_llm:
  phase2:
    api_url: "http://localhost:8000/api/optimize"
    timeout_seconds: 5
```

**Usage**: Orchestrator queries API for AI-driven optimization decisions

**Enable**:
```bat
cd C:\Users\User\home\vfx-deterministic-llm
python production_ready_system.py
```

### 2. MCP houdini-context Server

**Purpose**: Pattern learning for Houdini optimizations

**Configuration** (`config.yaml`):
```yaml
mcp:
  houdini_context:
    enabled: true
    tools:
      - "recall_similar_patterns"
      - "remember_hdk_pattern"
```

**Workflow**:
1. Houdini engineer queries MCP for similar scenes
2. MCP returns historical optimizations
3. Expert incorporates into recommendations
4. Successful optimizations stored back to MCP

### 3. ComfyUI VRAM Coordination

**Automatic detection**: Monitors `python.exe` with `ComfyUI\main.py`

**VRAM fencing**:
```yaml
vram_allocation:
  parallel:
    comfyui_gb: 2  # Limit ComfyUI to 2GB when other apps active
```

---

## 📊 Benchmarking & Regression Detection

### Synthetic Benchmarks

**Houdini Benchmarks** (`benchmarks/houdini_synthetic.py`):
- Karma XPU render (100M polygon scene)
- PDG/TOPs cook (100 work items)
- Vellum solve (500K points, 1M constraints)
- Pyro simulation (256³ voxel grid)

**UE5 Benchmarks** (`benchmarks/ue5_synthetic.py`):
- Lumen GI compilation
- Nanite streaming (1B triangles)
- VSM updates (10 dynamic shadows)
- Path Tracer convergence

**Run benchmarks**:
```bat
python benchmarks/houdini_synthetic.py
python benchmarks/ue5_synthetic.py
```

### Regression Detection

**Automated weekly testing**:
```yaml
regression_detection:
  test_frequency: "weekly"
  test_schedule_cron: "0 2 * * 0"  # 2 AM Sunday
```

**Statistical analysis**:
- Welch's t-test (p < 0.05)
- Effect size threshold: 10%
- Automatic rollback if regression detected

---

## 🧬 cuTensor + CUTLASS Acceleration

### Phase 2: Python SOP Acceleration

**Location**: `acceleration/cutensor_sop.py`

**Usage in Houdini**:
1. Add Python SOP to Vellum network
2. Reference `cutensor_sop.py`
3. Orchestrator detects scene complexity
4. Auto-enables cuTensor if beneficial

**Expected speedups**:
- Vellum constraint projection: **1.4x** (40% faster)
- Pyro advection: **1.3x** (30% faster)

**Requirements**:
```yaml
acceleration:
  cutensor:
    enabled: true
    precision_modes: ["fp32", "fp16", "fp8"]
    default_precision: "fp16"
```

### Phase 3: Custom CUTLASS Kernels (Future)

- Flash Attention for Lumen probe updates
- Custom GEMM kernels for Karma XPU
- Optimized tensor contractions for simulations

**Training TensorRT-LLM specialist**:
- Collect decision logs from Phase 2
- Fine-tune 1-2B parameter model
- Deploy via TensorRT-LLM for <5ms inference

---

## 📁 Project Structure

```
OptimizerV6/
├── config.yaml                     # Main configuration
├── requirements.txt                # Python dependencies
├── optimize_rtx4090_vfx.bat        # Main launcher
│
├── core/
│   └── gpu_optimizer_v6.py         # Orchestrator + 5 experts (~1100 lines)
│
├── monitoring/
│   ├── telemetry_monitor.py        # GPU metrics (pynvml, 100ms sampling)
│   └── vram_fence.py               # Per-app memory enforcement
│
├── dashboard/
│   ├── app.py                      # FastAPI backend
│   └── frontend/                   # React dashboard (build/)
│
├── integrations/
│   ├── vfx_llm_client.py           # vfx-deterministic-llm API client
│   └── mcp_houdini_client.py       # MCP houdini-context client
│
├── benchmarks/
│   ├── houdini_synthetic.py        # Houdini performance tests
│   ├── ue5_synthetic.py            # UE5 performance tests
│   └── regression_detector.py      # Statistical regression analysis
│
├── acceleration/
│   └── cutensor_sop.py             # cuQuantum Python SOP for Houdini
│
├── data/
│   ├── optimizer.db                # SQLite decision logs
│   ├── benchmarks.db               # Performance baselines
│   ├── telemetry.db                # Historical GPU metrics
│   └── nvidia_research.db          # Cached research patterns
│
└── docs/
    └── architecture.md             # Detailed system design
```

---

## 🎓 Expert Details

### Expert 1: NVIDIA GPU Researcher

**Sources**:
- nvidia/cutlass (GitHub)
- nvidia/TensorRT-LLM (GitHub)
- nvidia/cuQuantum (GitHub)
- developer.nvidia.com/blog

**Patterns Extracted**:
- Ada Lovelace FP8 optimizations (2x throughput)
- Thread block clusters (sm_89 feature)
- Flash Attention kernels for attention ops
- cuTensor tensor network patterns

**Update Frequency**: Weekly (configurable)

### Expert 2: NVIDIA GPU Optimization Specialist

**Responsibilities**:
- CUDA environment variables (cache, memory)
- Driver settings (power mode, scheduler)
- VRAM pressure detection
- Kernel launch parameter tuning

**Key Optimizations**:
```
CUDA_CACHE_MAXSIZE=4GB → -5% shader compilation time
PowerMizerMode=MaxPerf → -8% clock throttling
VRAM fencing → Zero OOM crashes
```

### Expert 3: Epic UE5.6 Engineer

**Console Variables Optimized**:
- **Lumen**: `r.Lumen.HardwareRayTracing`, `r.Lumen.Reflections.ScreenTraces`
- **Nanite**: `r.Nanite.Streaming.NumInitialRootPages=512` (12GB pool)
- **VSM**: `r.Shadow.Virtual.MaxPhysicalPages=4096`
- **Path Tracer**: `r.PathTracing.SamplesPerPixel=256`, `r.PathTracing.Denoiser=1`

**Integration**: Modifies `Engine.ini` dynamically

### Expert 4: SideFX Houdini 21 Engineer

**Optimizations**:
- **Karma XPU**: Samples, denoiser, ray depth
- **PDG/TOPs**: GPU scheduling, max concurrent cooks
- **Vellum/Pyro**: cuTensor acceleration detection
- **Environment**: `HOUDINI_OCL_MEMSIZE_MB`, `HOUDINI_VDB_FAST_TEMP_FILE`

**MCP Integration**: Recalls similar scene patterns, stores successful opts

### Expert 5: System Integration Specialist

**System-Level Checks**:
- PCIe Gen4 x16 validation
- Resizable BAR enabled
- Windows GPU Scheduler (HwSchMode=2)
- Thermal monitoring (83°C warning, 90°C critical)

**Multi-App Coordination**:
- Detects parallel workflows
- Enforces VRAM fencing (10GB + 10GB + 2GB + 2GB = 24GB)
- Adjusts process priorities

---

## 🐛 Troubleshooting

### Issue: "Python not found"

**Solution**:
```bat
REM Add Python to PATH
set PATH=C:\Users\User\AppData\Local\Programs\Python\Python313;%PATH%
```

### Issue: "CUDA not detected"

**Solution**:
```bat
REM Verify CUDA installation
nvcc --version

REM If missing, install CUDA Toolkit 12.6+
REM https://developer.nvidia.com/cuda-downloads
```

### Issue: "cuQuantum installation fails"

**Solution**:
```bat
REM cuQuantum requires CUDA 12.x
pip install cuquantum-python-cu12 --extra-index-url https://pypi.nvidia.com
```

### Issue: "Dashboard not accessible"

**Solution**:
```bat
REM Check if FastAPI is running
netstat -an | findstr :8090

REM Manually start dashboard
python dashboard/app.py
```

### Issue: "High GPU temperature warnings"

**Solution**:
1. Check GPU fan curve in MSI Afterburner / NVIDIA Control Panel
2. Verify case airflow
3. Reduce quality settings via config.yaml:
   ```yaml
   baselines:
     houdini:
       karma_xpu_render:
         samples: 128  # Reduce from 256
   ```

### Issue: "VRAM crashes in parallel mode"

**Solution**:
```yaml
vram_allocation:
  fencing:
    hard_limit_percent: 90  # Reduce from 95%
```

---

## 📈 Performance Validation

### Before OptimizerV6

| Workload | Time | Issues |
|----------|------|--------|
| Karma XPU (100M poly) | 56s | Occasional shader recompilation |
| PDG/TOPs (100 items) | 141s | GPU underutilized |
| Vellum (5M constraints) | 3.3s/frame | CPU bottleneck |
| Lumen GI rebuild | 240s | Software raytracing |
| Nanite (1B tris) | 15 hitches/min | Small streaming pool |

### After OptimizerV6 (Projected)

| Workload | Time | Improvement |
|----------|------|-------------|
| Karma XPU | **45s** | **-20%** (CUDA cache, OptiX opts) |
| PDG/TOPs | **120s** | **-15%** (GPU scheduling) |
| Vellum (w/ cuTensor) | **2.5s/frame** | **-25%** (Tensor acceleration) |
| Lumen GI rebuild | **180s** | **-25%** (HW raytracing enabled) |
| Nanite (1B tris) | **0 hitches/min** | **-100%** (12GB streaming pool) |

---

## 🔮 Roadmap

### Phase 1: Foundation (Complete)
- ✅ Core orchestrator with 5-expert MoE
- ✅ Tier 1 environment variable optimizations
- ✅ Configuration system
- ✅ Main launcher script

### Phase 2: Production Integration (In Progress)
- 🚧 Telemetry monitor (pynvml + WebSocket)
- 🚧 Web dashboard (FastAPI + React)
- 🚧 vfx-deterministic-llm integration
- 🚧 MCP houdini-context integration
- 🚧 Synthetic benchmarks (Houdini + UE5)
- 🚧 Regression detection (SQLite + Welch's t-test)
- 🚧 cuTensor Python SOP prototype

### Phase 3: Research Components (Future)
- ⏳ Custom CUTLASS kernels (Flash Attention for Lumen)
- ⏳ Fine-tuned TensorRT-LLM specialist model
- ⏳ HDK plugins (cuTensor Vellum solver)
- ⏳ Custom OptiX shaders (Karma XPU optimizations)

---

## 📚 References

### NVIDIA Research
- [CUTLASS 3.x Documentation](https://github.com/NVIDIA/cutlass)
- [TensorRT-LLM Guide](https://github.com/NVIDIA/TensorRT-LLM)
- [cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk)
- [Ada Lovelace Architecture Whitepaper](https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/)

### Houdini Optimization
- [Karma XPU Documentation](https://www.sidefx.com/docs/houdini/render/karma.html)
- [PDG/TOPs GPU Scheduling](https://www.sidefx.com/docs/houdini/tops/intro.html)
- [Vellum Solver Internals](https://www.sidefx.com/docs/houdini/nodes/dop/velumsolver.html)

### Unreal Engine Optimization
- [Lumen Technical Guide](https://docs.unrealengine.com/5.0/en-US/lumen-technical-details-in-unreal-engine/)
- [Nanite Virtualized Geometry](https://docs.unrealengine.com/5.0/en-US/nanite-virtualized-geometry-in-unreal-engine/)
- [Virtual Shadow Maps](https://docs.unrealengine.com/5.0/en-US/virtual-shadow-maps-in-unreal-engine/)

---

## 📝 License

**MIT** — see [LICENSE](LICENSE).

The harness is MIT because the gates are only useful if other people
measuring their own cards can read them and reuse them. Under default
copyright they could read them and not build on them.

The measurements in `measurements/` are real results from one machine
(an RTX 4090 and a Threadripper PRO 7965WX). The *method* is portable; the
numbers are not, and the records say which is which.

---

## 🙏 Acknowledgments

- **NVIDIA Research** for CUTLASS, TensorRT-LLM, cuQuantum
- **SideFX** for Houdini 21 HDK
- **Epic Games** for Unreal Engine 5.6 source access
- **Anthropic** for Claude Code assistance in system development

---

## 📧 Contact

For issues or questions:
- Check `data/optimizer.log` for detailed error messages
- Review configuration in `config.yaml`
- Consult expert reasoning with `verbose_experts: true`

---

**OptimizerV6 - Pushing RTX 4090 to its limits for professional VFX workflows**

*Generated for Joseph Ibrahim's VFX Workstation - Research + Production Hybrid*
