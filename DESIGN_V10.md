# Optimizer V10 — Ada-Native Low-Precision Serving

| Field | Value |
|---|---|
| **Disposition** | **Additive extension of V9, not a rewrite.** New backends + a whole-pipeline benchmark slot onto V9's existing `AccelerationBackend` ABC, `Action` enum, advisor framework, cache, and Pareto gate. V9 stays intact and is the rollback. |
| **Authored** | 2026-06-24, ARCHITECT role. Gate before any installs touch the machine. |
| **Target HW (DETECTED, not hardcoded)** | RTX 4090 **sm_89 (Ada Lovelace)**, 24 GB GDDR6X, 1.01 TB/s, ReBAR on · Threadripper PRO 7965WX 24c/48t · 128 GB DDR5 · **Studio driver 610.62** · Win11 Pro WS · 6K Apple Pro Display XDR. |
| **Primary workload** | ComfyUI CLI/server — Flux / Cosmos / Wan / video-DiT (24 GB VRAM is the binding constraint). Preserved: Houdini 21 (Karma XPU / Vellum / Pyro), UE5.6+. |
| **Research basis** | Web-verified 2026-06-24; recorded in memory `v10-research-findings`. |

---

## 1. The first-principles constraint that bounds everything

**The 4090 is Ada (sm_89), and the silicon is frozen.** This is the honest negative space — it kills "next-gen NVIDIA" marketing that only lands on Blackwell before we spend a line of code on it.

**Do NOT chase on this card (Blackwell sm_100/sm_120 only):**
- FP4 / NVFP4 / MXFP4 and the 2nd-gen Transformer Engine.
- DLSS 4 *multi*-frame generation, 5th-gen tensor cores.

**What this card actually has:** FP8 (E4M3/E5M2) 4th-gen tensor cores, INT4 tensor path, Shader Execution Reordering, Optical Flow Accelerator (single frame-gen + DLSS-4 transformer SR/RR — these *do* run on Ada), ReBAR, PCIe Gen4 x16.

> The lever on a 4090 in 2026 is **not new silicon** — it is the 2025–26 software wave that targets exactly Ada **INT4 + FP8** and the **24 GB VRAM wall**.

## 2. The inversion (the load-bearing design decision)

A naive "use NVIDIA's next-gen FP8 everywhere" plan is **wrong for this card.** NVIDIA's own docs state *"Ada GPUs support FP8 with reduced efficiency"* (FP8 is GEMM-only on cc 8.9). The measured diffusion precision hierarchy on Ada is:

> **INT4 (SVDQuant / Nunchaku, ~3× + 3.6× VRAM cut)  >  FP8 (modelopt / TRT, reduced efficiency on Ada)  >  BF16**

Therefore V10's **headline lever is INT4 SVDQuant serving**, and FP8/TRT is the **fallback** for models Nunchaku does not quantize — not the reverse. Sourced: Nunchaku v1.2.1 (Ada = INT4 only; NVFP4 is Blackwell-exclusive); Flux-schnell 17 GB → ~6 GB, ~3× over BF16.

## 3. What V10 MUST NOT change (V9 invariants, preserved verbatim)

1. **Detect, don't hardcode.** All HW facts via `HardwareProfile`. Zero driver/arch literals. (V10 adds no constants — `sm_89`, INT4-vs-FP4 selection, and the Py-version split are all *derived* from the profile, not written down.)
2. **Degrade, never crash.** Every new backend probes its SDK in `__init__`, sets `enabled=False` + `_unavailable_reason` on ImportError; `compile()` raises a clear install hint. One dead backend never breaks `once`/`watch`.
3. **Off-OneDrive for anything mutable** (`paths.py`).
4. **Honest capability.** `doctor` reports new backends truthfully (the V9 sin to avoid: Sage reporting "v2/ENABLED" while v1 was installed).
5. **Prove it or don't claim it.** Every speedup is `benchmark`-measured and Pareto-gated, or it's labeled a prior. **This is the spine V10 extends, not replaces.**
6. **One orchestrator, one `Action` enum, one config.**

## 4. Environment contract (REVISED 2026-06-24 after on-machine recon)

The blueprint assumed a fresh Python 3.13 venv. **Recon overrode that — for the better.** The existing `D:\OptimizerV9\venv` is already **Python 3.12.10 + torch 2.8.0+cu126 + TensorRT 11.0.0.114 (CUDA available) + CUTLASS 4.5.1** — the entire expensive stack. The only Python 3.13 on the box is the **Microsoft Store** build (a poor base for native CUDA extensions), and the cp314-TRT problem (NVIDIA/TensorRT #4725 still open) only ever applied to the *system* 3.14 interpreter — never this venv.

| Interpreter | Role | Rationale |
|---|---|---|
| **`D:\OptimizerV9\venv` — Python 3.12.10** (REUSE) | The acceleration venv: torch 2.8 cu126, TRT 11, CUTLASS 4.5.1, **triton-windows (added)**, Sage, +Nunchaku next. | Already proven; 3.12 has TRT / triton-windows / Sage / Nunchaku wheels. Avoids re-downloading multi-GB and the Store-Python risk. |
| `C:\Python314` (system) | Config-tuning only (advisors, env/registry, doctor). | No TRT wheels on 3.14; fine for the side-effect family. |

Discipline: **only ADD packages to this venv, never upgrade/remove the working torch/TRT.** Rollback is the source tree + intact V9 (per V9's own principle), and each addition is individually `pip uninstall`-reversible. V10 keeps V9's off-OneDrive data root (`D:\OptimizerV9`) and `paths.py`; a separate root is unnecessary now the venv is shared. A thin `v10_optimizer.bat` fronts the same package as the new entry point. (cu126 vs the cu128 Sage-2.x prebuilt wheels is the one ABI item to resolve when upgrading Sage — see §9.)

## 5. The five new backends (against the real V9 `AccelerationBackend` ABC)

Each subclasses `accel/backend.py:AccelerationBackend`, is constructed via `registry.register_factory` (so a broken SDK is a dead entry, never a crash), and appears in `doctor`'s capability map. New `artifact_format` strings extend the existing `{trt_plan, cutlass_so, noop}` set.

| # | Backend (new file) | `artifact_format` | `applies_to` | Ada speedup (prior, to be measured) | Quality Δ | Install cost |
|---|---|---|---|---|---|---|
| **1** | `accel/sage.py` — **activate** existing | `noop`* | attention, any diffusion shape | ~1.3–1.5× SDPA | ~0 (lossless attn quant) | triton-windows + Sage wheel |
| **2** | `accel/torchcompile.py` (new) | `torch_compiled` | unet/dit, static-ish shapes | ~1.2–1.4× | 0 (numerically identical) | **none** (live torch 2.8) |
| **3** | `accel/nunchaku.py` (new) — **headline** | `nunchaku_int4` | flux/dit, W4A4-quantizable | **~3× + 3.6× VRAM** | low (gated) | Nunchaku 1.2.1 + ComfyUI-nunchaku |
| **4** | `accel/modelopt_trt.py` (new) | `trt_plan` (reuses TRT loader) | unet/dit when not INT4-quantizable | ~1.5–1.8× (FP8, Ada-derated) | low (gated) | modelopt + existing TRT |
| **5** | `accel/cutlass.py` — **existing**, kept | `cutlass_so` | gemm (Houdini Vellum/Pyro) | 1.89× (already MEASURED) | ~0 | already activated |

\* Sage activates ComfyUI's SDPA patch (a launch-flag / custom-node side-effect), so it reports through the capability map + the ComfyUI advisor rather than emitting a compiled artifact; it has no `compile()` output of its own. Its `enabled` flag must reflect **triton importable AND Sage ≥ 2.0** — the honest check that V9 got wrong.

**Backend selection order** (encodes the §2 inversion, lives in the ComfyUI advisor / coordinator, not in any backend): for a diffusion model — try **Nunchaku INT4** → else **modelopt FP8 TRT** → else **torch.compile** → always **Sage** for attention on top. Houdini Vellum/Pyro → **CUTLASS FP8 GEMM** (unchanged).

## 6. Whole-pipeline benchmark (extends `benchmark.py`, the real proof)

V9's `benchmark` proves a *GEMM microbench* (matmul + CUTLASS FP8). It cannot yet decide a diffusion tradeoff because the metric is wrong. V10 adds an **end-to-end harness** that measures what actually matters and fills the quality columns that are currently `0.0` placeholders.

- **New metric, per (model, precision, backend):** `seconds_per_image` (Flux/SDXL) and `seconds_per_frame` (Wan/Cosmos/video-DiT), plus `vram_peak_gb` (the 24 GB wall is a first-class admit/reject axis on this card).
- **Real quality deltas:** run a fixed seed/prompt set through the baseline and the accelerated path; compute **LPIPS** and **CLIP** deltas (the `quality.record_quality(lpips_delta, clip_delta, ...)` signature already exists — V9 passes 0.0; V10 passes real values).
- **Same Pareto gate, now meaningful:** `speedup ≥ MIN_SPEEDUP AND lpips ≤ MAX_LPIPS_DELTA AND clip ≥ MIN_CLIP_DELTA AND vram_peak ≤ budget`. The gate already *refused* a losing CUTLASS kernel — V10 lets it refuse a fast-but-ugly INT4 quant the same way.
- **Isolation:** the image run executes in a subprocess (mirrors `_run_fp8_probe`) so a bad CUDA launch can't crash the harness; absent torch/CUDA → reports honestly, writes nothing fake.

## 7. Advisor deltas (surgical, against the existing rules)

- **`advisors/comfyui.py`** — R1 (Sage) becomes a *real* activation path (triton-windows present? flag set? node installed?) instead of a flat "~1.5x" prior. **New R5: recommend the Nunchaku INT4 path** for a hot Flux/DiT model when (a) the model has an INT4 SVDQuant build available and (b) the pipeline benchmark admitted it — emits `ENABLE_BACKEND` / `INSTALL_CUSTOM_NODE` (`nunchaku-tech/ComfyUI-nunchaku`). FP8-via-modelopt stays as the existing R3 speculative-TRT path, now correctly positioned as the fallback.
- **`advisors/houdini.py`** — close V9's stated gap: wire the proven FP8 GEMM into a live Vellum/Pyro solve SOP (`WIRE_BACKEND_PATCH`), per-shape autotune cache.
- **`advisors/ue5.py`** — add Ada-valid cvar advice: DLSS-4 transformer SR/RR plugin, RTX Mega Geometry, Shader Execution Reordering. (Config family — instant, reversible.)
- **No new app coupling in `context.py`** — advisors collect their own state, per the frozen contract.

## 8. Phasing / mile markers

- **Mile 1 — Blueprint (this doc).** Gate. ✓
- **Mile 2 — Acceleration venv. DONE via REUSE (2026-06-24).** Recon found the existing 3.12 venv already carries torch/TRT/CUTLASS; no new venv built. `doctor` green on it.
- **Mile 3 — Quick wins (zero/low install). ✓ DONE (2026-06-24).**
  - *SageAttention:* woke from dormant; Sage **1.0.6** runs clean (maxdiff ≤0.006), **measured 2.34× @ seq4096 / 0.33× @ seq1024**; version-honesty bug fixed in `accel/sage.py`.
  - *torch.compile:* new `accel/torchcompile.py` (max-autotune + CUDA graphs, off-OneDrive Inductor cache), wired into the registry. **Proven on the 4090** — numerically lossless (maxdiff 0.0039), 7.5× on a micro-block (launch-overhead-bound; real UNet ~1.3–2×, Mile 4 will measure true). Four backends now ENABLED (TRT, Sage, torch_compile, CUTLASS). Suite 101 passed / 3 skip.
  - *Env lesson (encoded):* **triton-windows must match torch** — torch 2.8 ↔ **triton 3.4.x** (the installed 3.7.1 broke Inductor: missing `triton_key`; `has_triton`→False). Pinned to `triton-windows==3.4.0.post21`. Windows also needs the Inductor **static CUDA launcher disabled** (LLP64 C-long overflow) — handled in `torchcompile.py`.
  - *Remaining:* Sage **v2** upgrade for FP8/INT8 attn (cu126 ABI, §9) — optional, deferred to the heavy-install pass with Nunchaku.
- **Mile 4 — Pipeline benchmark. ✓ DONE (synthetic path, 2026-06-24).** New `v9/pipeline_benchmark.py` + `pbench` CLI subcommand: a diffusion-shaped DiT/Flux block stack (attention via SDPA so Sage applies; whole-graph so torch.compile applies), measuring **latency, vram_peak, and tensor-fidelity (rel_err)** per backend, Pareto-gated, persisted to the quality+telemetry ledgers. Measured the real **Sage crossover end-to-end**: seq1024 0.98× (reject) → seq4096 1.24× (reject) → **seq8192 1.37× ADMIT** (rel_err 6e-4). The gate both rejects sub-threshold and admits real wins. Honest dilution surfaced: Sage's full-block gain (1.24–1.37×) is lower than the isolated-attention microbench (2.34×) because MLP/norms aren't swapped — exactly why the pipeline metric matters. Suite 102 passed / 3 skip.
- **Mile 4b — REAL-image LPIPS. ✓ DONE (2026-06-24).** New `v9/image_benchmark.py` + `ibench` CLI subcommand: drives a real **SD1.5** generation through diffusers at a fixed seed, measures **seconds/image, vram_peak, and REAL LPIPS** (no more placeholder), saves images off-OneDrive. Measured on the 4090: **torch.compile 2.05–2.15× at LPIPS 0.0005–0.002 → PARETO ADMIT: YES** (perceptually identical). Honest finding: **SageAttention 1.0.6 is n/a for SD1.5** (head_dim 40; Sage needs 64/96/128 → it applies to SDXL/Flux/video, not SD1.5). Suite 103 passed / 3 skip. CLIP score is the one remaining quality axis (needs open-clip) — clip carried as 0.0, not faked.
- **Mile 5 — Nunchaku INT4 backend + ComfyUI R5.** The headline. Prove ~3× / 3.6×-VRAM on *this* rig or the gate rejects it.
- **Mile 6 — modelopt→TRT FP8 fallback + Houdini Vellum wire + UE5 cvars + Crucible adversarial pass** (degrade-safe, no-stale-constant, `once` exits 0, regression vs V9).

Dependency order: **venv → {Sage ‖ torch.compile} → pipeline-benchmark → Nunchaku → {modelopt ‖ Houdini ‖ UE5} → Crucible.**

## 9. Open items to resolve at build time (not blocking the blueprint)

- Nunchaku model coverage beyond Flux for **Wan / Cosmos video-DiT** — verified for Flux family; confirm the others have SVDQuant INT4 builds, else those stay on the FP8/TRT fallback.
- Sage 2.2.0 wheel is published as **cu128/torch2.8**; confirm ABI against the chosen cu128 venv (or build the cu126 variant) at install.
- torch.compile + ComfyUI: prefer `mode="max-autotune"` + CUDA graphs; verify graph-break behavior on dynamic diffusion shapes (may need `dynamic=True` or per-shape compile cache).
- **Install discipline (learned the hard way 2026-06-24):** a plain `pip install diffusers transformers` **clobbered torch 2.8.0+cu126 → 2.12.1+cpu** (resolver pulled CPU torch), breaking CUDA/Sage/TRT. ALWAYS install ML deps with care: pin torch, use `--no-deps` for restores, and re-verify `torch.cuda.is_available()` after any install. Restore = `pip install --no-deps --force-reinstall torch==2.8.0 torchvision==0.23.0 --index-url …/cu126`.
- **transformers must be 4.x for diffusers 0.38 single-file** — transformers 5.x refactored `CLIPTextModel` (no `.text_model`), breaking `from_single_file`. Pinned `transformers==4.49.0` (tokenizers 0.21.4). torch↔torchvision: 2.8.0 ↔ 0.23.0.

---
*Gate: confirm the inversion (§2), the env split (§4), and the backend set (§5). Then Mile 2 fires.*
