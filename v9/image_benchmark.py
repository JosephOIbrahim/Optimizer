"""V10 — Mile 4b: REAL-image pipeline benchmark with measured LPIPS.

Where `pipeline_benchmark.py` proves the backends on a diffusion-shaped synthetic
module (tensor-fidelity), this runs an actual **Stable Diffusion** generation
through diffusers and measures the numbers that decide a real tradeoff:

  * seconds/image      — wall time for a full denoise;
  * vram_peak_gb       — peak allocated VRAM;
  * lpips_delta        — REAL perceptual distance (LPIPS-alex) between the
                         accelerated image and the eager baseline at the SAME
                         seed. This is the value the Pareto gate was always
                         meant to use — no more 0.0 placeholder.

Backends measured: torch.compile (wraps the UNet) and SageAttention (patches the
UNet's SDPA). Same fixed seed across runs so LPIPS reflects only the backend's
numerical effect, not sampling noise. Images are written off-OneDrive so the
result is inspectable.

Honest/degrade-safe: missing torch/CUDA/diffusers/model -> says what it can't do
and writes nothing fake.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from v9 import paths, quality, telemetry
from v9.config import Config, load_config

_DEFAULT_MODEL = r"C:\ComfyUI\models\checkpoints\v1-5-pruned-emaonly.safetensors"
_DEFAULT_PROMPT = ("a cinematic photograph of a weathered lighthouse on a cliff "
                   "at golden hour, dramatic clouds, highly detailed")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")


def _probe():
    try:
        import torch  # type: ignore
        import diffusers  # type: ignore
        import lpips  # type: ignore
        return torch, diffusers, lpips
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def _load_pipe(torch: Any, diffusers: Any, model_path: str):
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        model_path, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _gen(torch: Any, pipe: Any, prompt: str, steps: int, size: int, seed: int):
    """One generation at a fixed seed; returns (PIL image, seconds, vram_gb)."""
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = pipe(prompt, num_inference_steps=steps, height=size, width=size,
               generator=g, guidance_scale=7.5).images[0]
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    return img, secs, vram


def _lpips_delta(torch: Any, lpips_mod: Any, img_a, img_b) -> float:
    """Real LPIPS-alex between two PIL images in [0,1]->[-1,1]."""
    import numpy as np
    loss = getattr(_lpips_delta, "_net", None)
    if loss is None:
        loss = lpips_mod.LPIPS(net="alex", verbose=False).to("cuda")
        _lpips_delta._net = loss

    def to_t(im):
        a = np.asarray(im.convert("RGB"), dtype="float32") / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to("cuda")
        return t * 2.0 - 1.0
    with torch.no_grad():
        return float(loss(to_t(img_a), to_t(img_b)).item())


def run(config: Optional[Config] = None, model_path: str = _DEFAULT_MODEL,
        prompt: str = _DEFAULT_PROMPT, steps: int = 25, size: int = 512,
        seed: int = 1234, tc_mode: str = "default",
        backends=("torch_compile", "sage_attention_v2")) -> int:
    if config is None:
        config = load_config()
    model_path = model_path or _DEFAULT_MODEL    # CLI may pass None -> default
    prompt = prompt or _DEFAULT_PROMPT

    print("=" * 72)
    print(" Optimizer V10 — REAL-image benchmark (SD1.5; measured LPIPS)")
    print("=" * 72)
    print("  " + config.profile.summary())

    torch, diffusers, lpips_or_err = _probe()
    if torch is None:
        print(f"\n  missing dependency — {lpips_or_err}. (0 rows, nothing faked.)")
        return 0
    if not torch.cuda.is_available():
        print("\n  no CUDA — refusing to benchmark a diffusion pipeline on CPU.")
        return 0
    lpips_mod = lpips_or_err
    if not Path(model_path).exists():
        print(f"\n  model not found: {model_path}")
        return 0

    out_dir = paths.DATA_ROOT / "image_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  model: {model_path}")
    print(f"  prompt: {prompt!r}")
    print(f"  steps={steps} size={size} seed={seed} | out={out_dir}")

    print("\n  loading SD1.5 from single-file checkpoint ...")
    try:
        pipe = _load_pipe(torch, diffusers, model_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  pipeline load FAILED: {type(exc).__name__}: {exc}")
        return 1

    from v9.accel.registry import build_default_registry
    from v9.accel.torchcompile import TorchCompileBackend
    reg = build_default_registry(config.profile, config)

    # ---- baseline (eager) ------------------------------------------------
    print("  generating baseline (eager) ...")
    base_img, base_s, base_vram = _gen(torch, pipe, prompt, steps, size, seed)
    base_path = out_dir / "baseline.png"
    base_img.save(base_path)
    print(f"    baseline: {base_s:.2f}s  vram={base_vram:.2f}GB  -> {base_path}")

    header = (f"  {'backend':22s} {'sec/img':>8s} {'speedup':>8s} "
              f"{'vram_gb':>8s} {'lpips':>7s} {'admit':>6s}")
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    rows = 0
    for name in backends:
        try:
            # restore a clean eager UNet each round
            if name == "torch_compile":
                tc = TorchCompileBackend(mode=tc_mode)
                if not tc.enabled:
                    print(f"  {name:22s}  skipped (disabled)")
                    continue
                orig_unet = pipe.unet
                pipe.unet = tc.load(None).runner.wrap(pipe.unet)
                # WARMUP: first call triggers the (one-time, cached) compile — not
                # timed. The timed run below is steady-state, the honest number.
                _gen(torch, pipe, prompt, steps, size, seed)
                img, secs, vram = _gen(torch, pipe, prompt, steps, size, seed)
                ver = tc.version
                pipe.unet = orig_unet          # restore eager UNet for next round
            elif name == "sage_attention_v2":
                sage = reg.get("sage_attention_v2")
                if sage is None or not sage.enabled:
                    print(f"  {name:22s}  skipped (disabled)")
                    continue
                img, secs, vram = _gen_with_sage(
                    torch, sage, pipe, prompt, steps, size, seed)
                ver = sage.version
            else:
                continue
        except AssertionError as exc:
            # SageAttention rejects unsupported head dims (SD1.5 = 40; Sage needs
            # 64/96/128). Honest n/a, not a crash.
            print(f"  {name:22s}  n/a  ({str(exc)[:60]})")
            continue
        except Exception as exc:  # noqa: BLE001 - one backend can't sink the run
            print(f"  {name:22s}  FAILED ({type(exc).__name__}: {str(exc)[:50]})")
            continue

        speedup = base_s / secs if secs > 0 else 1.0
        lp = _lpips_delta(torch, lpips_mod, img, base_img)
        img_path = out_dir / f"{name}.png"
        img.save(img_path)

        verdict = quality.record_quality(
            ts_utc=_utc_now_iso(), speedup=speedup,
            lpips_delta=lp, clip_delta=0.0,   # REAL lpips now; clip later
            app="comfyui", backend_name=name, backend_version=ver,
            model_sha256="sd15_single_file", shape_w=size, shape_h=size,
            batch=1, precision="fp16",
            notes={"harness": "image_sd15", "sec_per_img": secs,
                   "vram_peak_gb": vram, "steps": steps, "lpips_net": "alex",
                   "out": str(img_path)})
        telemetry.record_generation(
            ts_utc=_utc_now_iso(), workflow_hash="sd15_512", model_sha256="sd15",
            shape_w=size, shape_h=size, batch=1, sampler="dpm", steps=steps,
            precision="fp16", backend_set=[name], wall_seconds=secs,
            samples_per_sec=(1.0 / secs if secs else 0.0), vram_peak_gb=vram)
        rows += 1
        print(f"  {name:22s} {secs:8.2f} {speedup:8.3f} {vram:8.2f} "
              f"{lp:7.4f} {'YES' if verdict.admitted else 'no':>6s}")

    print(f"\n  wrote {rows} row(s) with REAL LPIPS to the quality ledger.")
    print(f"  Pareto gate: speedup>={quality.MIN_SPEEDUP} AND "
          f"lpips<={quality.MAX_LPIPS_DELTA} AND clip>={quality.MIN_CLIP_DELTA}")
    print(f"  images saved under {out_dir} — open baseline.png vs each backend.")
    return 0


def _gen_with_sage(torch: Any, sage: Any, pipe: Any, prompt: str, steps: int,
                   size: int, seed: int):
    """Generate with the SageAttention SDPA patch active for the whole denoise."""
    le = sage.load(None)
    with le.runner:                       # SageSDPAPatch context manager
        return _gen(torch, pipe, prompt, steps, size, seed)


if __name__ == "__main__":
    raise SystemExit(run())
