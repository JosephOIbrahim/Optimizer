# Optimizer V11 — A Hardware-Modeling, Self-Tuning Control Core

| Field | Value |
|---|---|
| **Disposition** | Re-architecture of the *decision half* of the optimizer. The V9 substrate (detect → cache → Pareto-gate, degrade-safe) is KEPT; the hand-written advisors + manual sweeps are REPLACED by a measurement-driven control loop. V9/V10 stay intact as rollback. |
| **Authored** | 2026-06-24, ARCHITECT role, after Joe confirmed the pivot. Gate before any code. |
| **Target HW (DETECTED)** | RTX 4090 sm_89 (Ada), 24 GB, 1.01 TB/s, 128 SMs · Threadripper PRO 7965WX 24c/48t · 128 GB · driver 610.62 · CUDA 13.2 toolchain. |
| **Basis** | First principles + web-verified NVIDIA research (2026-06-24): CUPTI/PyTorch-profiler, CuTe DSL, CUDA green contexts (runtime API, Ada), NVIDIA Warp-in-Houdini, roofline/speed-of-light-guided autotuning literature. |

---

## 1. The shift (why V11 exists)

V9+V10 is, honestly, a **rules-based config-tuner + acceleration-library integrator** wrapped around one good idea — the empirical `measure → Pareto-gate → cache` loop. It is not a next-gen optimizer because **it has no model of the hardware and no observability into the real workload.** Every "optimization" is a hand-authored prior validated *after the fact*.

An optimizer is a **closed-loop controller over a configuration space toward an objective.** Five parts; V9's scorecard:

| Component | V9 state | V11 |
|---|---|---|
| Objective | speedup only | + perf/watt, vram_peak, throughput-under-contention |
| **Hardware model** | **absent** | **roofline / speed-of-light, derived from detected HW** |
| **Observability** | `nvidia-smi` util% only | **CUPTI / PyTorch-profiler kernel metrics** |
| **Search** | one *manual* CUTLASS sweep; `genetic_tuner` is a stub | **roofline-guided autotuner over CuTe DSL candidates** |
| Admission + cache | **present, good** | kept verbatim |

The two missing rows are precisely what make an optimizer "next-gen," and precisely what NVIDIA Labs research is about.

## 2. The loop (first principles)

```
        ┌────────────────────────────────────────────────────────┐
        │                                                        ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ OBSERVE  │──▶│  MODEL   │──▶│  DECIDE  │──▶│   ACT    │──▶│  ADMIT   │
  │ profile  │   │ roofline │   │ search   │   │ actuator │   │ Pareto   │
  │ (CUPTI)  │   │ bound?   │   │ (tune)   │   │ (backend)│   │  + cache │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
   measured        speed-of-      pick the       apply the      keep only
   FLOPs/bytes     light ceiling  optimization   selected       what's BOTH
   /occupancy      + bound class  that moves     config         faster AND
                                  THAT bound                    not-worse
```

The optimizer **observes the real workload, places it against the hardware ceiling, derives which optimization class moves the actual bottleneck, searches that class, and admits empirically.** No guessing.

## 3. The hardware model (the missing foundation)

`hwmodel.py` extends `HardwareProfile` with a **roofline derived from detected hardware** (not hardcoded marketing TFLOPS — consistent with the V9 detect-don't-hardcode rule), then **validated by a max-throughput microbench** at startup:

- **Peak compute** per precision = (SM count) × (tensor/FP ops per SM per clock) × (detected boost clock). Validated against a measured GEMM at each precision (we already have an FP8 GEMM that hit 1.94×—does it hit speed-of-light? The model will say).
- **Peak DRAM BW** = detected (1.01 TB/s), validated by a streaming-copy microbench.
- **Ridge points** (arithmetic intensity where compute ≈ memory bound), order-of-magnitude for the 4090: **~82 (FP32) · ~164 (FP16-tensor) · ~327 (FP8-tensor) FLOP/byte** (dense; sparsity ~2×). The model computes these from the validated peaks.

**Why this matters, concretely:** almost every VFX/diffusion kernel runs at arithmetic intensity *well below* these ridges → **memory-bound** → the winning move is *cutting bytes* (fusion, quantization), which the model **predicts a priori** instead of discovering by trial. That single fact reorganizes the whole advisor layer from heuristics into derivations.

## 4. Observability

`profiler.py` — wrap the **PyTorch profiler (Kineto→CUPTI, already in torch 2.8)** for torch workloads (ComfyUI), and **Nsight Compute `ncu --section SpeedOfLight_RooflineChart`** / CUPTI for deep dives and non-torch workloads (Houdini/UE5 via trace + NVML coarse sampling). Emits a typed `KernelProfile{flops, dram_bytes, achieved_tflops, occupancy, duration_us}` and a rolled-up `WorkloadProfile`. Degrade-safe: no profiler → fall back to NVML coarse counters and say so.

*(Watch: the new Python-native profilers — Nsight Python / Tachyon — may replace the ncu shell-out; evaluate at build.)*

## 5. Decide — roofline-guided advisor + autotuner

- `roofline.py` places each `KernelProfile` on the model and classifies the bound, emitting the **optimization class** that moves it:
  - **memory-bound** → fuse (torch.compile) · quantize to cut bytes (fp8/int4) · better access pattern.
  - **compute-bound** → higher-throughput precision (tensor cores) · better tile.
  - **latency/occupancy-bound** → CUDA graphs · larger batch · more concurrency (green contexts).
- `autotune.py` searches the chosen class. **CuTe DSL (`nvidia-cutlass-dsl`)** generates+compiles kernel candidates at **30–100× the rate of per-config `nvcc`** — so the manual FP8-GEMM sweep becomes a real search over many more configs, cheaply. Prune with an analytical/learned cost model (cf. SynPerf), measure top-k, admit via the existing Pareto gate, cache per (shape, arch, driver). The dormant `genetic_tuner` is the seed of the search policy.

## 6. Act — actuators (backends become *selectable actions*)

The V10 inference backends (TRT, Sage, torch.compile, CUTLASS) are **demoted from "the product" to actuators the search selects.** New system-level actuators:

- **Green contexts** (`cuDevSmResourceSplit`, runtime API, Ada-supported): partition the 128 SMs so a Houdini sim and a ComfyUI inference co-run without clobbering — the *dynamic* replacement for static "VRAM fencing." Composes with MPS for multi-process.
- **DVFS / power policy** (NVML `nvmlDeviceSetGpuLockedClocks`, power-limit): find the perf/watt sweet spot for sustained Karma XPU renders; thermal-aware on long jobs (relevant with the 6K display + long renders).
- **Warp** (`warp-lang`, Houdini HDA): the VFX-native sim substrate — Vellum/Pyro/particle solves as differentiable Warp kernels, the legitimate version of the bespoke CUTLASS-Vellum-GEMM.

## 7. Phasing (mile markers)

- **V11-Mile 1 — Hardware model + roofline. ✓ DONE + ADVERSARIALLY VERIFIED (2026-06-24).** `hwmodel.py`: derives the 4090 roofline from detected clocks/cores (theoretical pinned to MAX clock = stable SoL), validated by microbench (fp32 92/102, fp16 166, fp8 336 TFLOPS; DRAM 930/1008 GB/s). `classify()` returns a structured `ClassifyResult`. **FP8 GEMM measured at 88% of speed-of-light → 12% headroom.** A 4-agent adversarial pass (0 critical, 7 major) confirmed the physics and drove fixes (clock-decoupling, ridge-absent guard, SoL-vs-measured, structured contract). 12 tests. `roofline` CLI.
- **V11-Mile 2 — Observability + 2b OI. ✓ DONE + HARDENED.** `profiler.py`: PyTorch-profiler → `KernelProfile` (achieved-TFLOPS + raw shapes). **Mile-2b: true operational intensity** — `_op_bytes` computes analytical DRAM bytes from op shapes (matmul read-operands+write-output; elementwise read-inputs+write-output), and `classify_workload` places the dominant op on the FULL roofline via `hwmodel.classify` (OI vs ridge + achieved compute AND bandwidth fractions). `classify_dominant` (compute-axis) is the fallback. Adversarially hardened (2 crit fixed): time-share gate (≥40% GPU time → no wrong-kernel), BALANCED hysteresis band (no noise-flip), shared `saturation_ceiling`, >1.1× = data-fault. Honest limit: bytes are *estimated* (min traffic; ignores L2 reuse — flagged; ncu `dram__bytes` = the measured upgrade).
- **V11-Mile 3 — Decide / closed loop. ✓ DONE + HARDENED (2 verify passes).** `control.py:analyze` = OBSERVE→MODEL→DECIDE; `recommend_from_bound` emits DERIVED levers with physically-correct, hedged rationales. `analyze` CLI. Live (OI-based, stable ×3): **gemm→COMPUTE** (OI 2731≫ridge 202, bw ~7%), **elementwise→MEMORY** (OI 0.2≪ridge, bw ~98%), **dit→BALANCED** (dominant flash-attention has no FLOP estimate → honestly inconclusive). Suite 132. *Hardening (final verify, 0 crit/4 major, all fixed):* OI is the **bound** axis; the achieved compute/bandwidth fractions add only the **saturation/latency** axis and are **NOT an independent check** (achieved_bw=bytes/time shares the OI byte estimate — real independent validation needs ncu `dram__bytes`). STEP-2 latency demotion confined to the near-ridge zone (fixes a cold-start COMPUTE↔LATENCY flip); `>1.1×` data-fault guard added to the OI path; headroom clamped ≥0.
- **V11-Mile 4 — Autotuner (lite). ✓ DONE (working).** `control.py:autotune` = roofline-guided precision search, on-device measured, **Pareto-gate admitted** (fp8 @ 2.12× selected + recorded to the quality ledger). `autotune` CLI. Full CuTe-DSL kernel-config search is the deeper version (deferred).
- **V11-Mile 5 — System actuators. ☐ DEFERRED.** Green-context SM partition + DVFS perf/watt. Clean next step; not faked.
- **V11-Mile 6 — Warp VFX core. ☐ DEFERRED.** Houdini Vellum/Pyro as Warp kernels. Substantial subsystem; documented, not faked.

Dependency: **hwmodel ✓ → profiler ✓ → {decide ✓ ‖ autotune ✓} → actuators (M5) → Warp (M6).** The control loop CLOSES today (analyze→autotune→gate→ledger); M5/M6 extend it.

## 8. Negative space (do NOT build)

- Datacenter research: MIG, NVLink fabric, multi-node, collective comms — irrelevant to one 4090.
- Blackwell-only: FP4/NVFP4, 2nd-gen Transformer Engine, DLSS-4 multi-frame-gen.
- More inference-QA harnesses (LPIPS/CLIP image grading) — that's application QA, not system optimization. `ibench` stays as a tool, doesn't grow.

## 9. Open items to verify at build

- Python-native profiler (Nsight Python / Tachyon) maturity vs the `ncu` shell-out for programmatic roofline.
- CuTe DSL on **Ada sm_89** specifically (most blog numbers are Blackwell) — confirm it compiles+runs our FP8 GEMM class on the 4090; it's public-beta until ~end of summer 2026.
- Green contexts from Python (driver-API ctypes vs a binding) and whether they help *across* processes here without full MPS setup.
- Houdini↔Warp HDA version + whether it exposes the solve we'd target.

---
*Gate: confirm the control-loop architecture (§2) and the phasing (§7). Then V11-Mile 1 (hardware model + roofline) fires — the foundation the whole loop stands on.*
