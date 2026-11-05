"""V10 — end-to-end pipeline benchmark (Mile 4).

Where `benchmark.py` proves a single GEMM microbench, this proves the *runtime*
backends (SageAttention, torch.compile) on a **diffusion-shaped** module — an
attention-heavy DiT/Flux-style transformer stack — measuring the numbers that
actually decide a diffusion tradeoff:

  * latency       — ms per forward (a seconds/image proxy);
  * vram_peak_gb  — peak allocated VRAM (the 24 GB wall is a first-class
                    admit/reject axis on this card);
  * fidelity      — relative L2 of the accelerated output vs the eager baseline
                    (a measured quality signal; perceptual LPIPS/CLIP need the
                    real image path and are recorded as deferred, not faked).

Honest like the rest of the system (DESIGN_V9 principle 5):
  * no torch / no CUDA -> says so, writes nothing.
  * each measured row is written to the quality + telemetry ledgers; the printed
    admit column gates on measured speedup AND fidelity; lpips/clip are carried
    as 0.0 placeholders with the real rel_err in notes until the image path
    lands (Mile 4b).

Two sequence lengths are run on purpose: SageAttention LOSES at short sequences
and WINS at long ones (the per-shape crossover), and the benchmark is where that
becomes a measured fact rather than a prior.
"""
from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from typing import Any, Optional

from v9 import quality, telemetry
from v9.config import Config, load_config

# tensor-space fidelity gate (perceptual LPIPS/CLIP come with the image path);
# a relative-L2 above this rejects the candidate as too lossy.
_REL_ERR_MAX = 0.05


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")


def _probe_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _build_dit_stack(torch: Any, dim: int, heads: int, depth: int):
    """A DiT/Flux-style transformer block stack: LayerNorm -> MHSA (via SDPA, so
    SageAttention's patch intercepts it) -> MLP, residual, x depth. This is the
    shape a diffusion UNet/DiT actually spends its time in."""
    import torch.nn as nn
    import torch.nn.functional as F

    class DiTBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            self.heads = heads

        def forward(self, x):  # x: (B, S, dim)
            b, s, d = x.shape
            h = self.norm1(x)
            qkv = self.qkv(h).reshape(b, s, 3, self.heads, d // self.heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, S, Dh)
            q, k, v = qkv[0], qkv[1], qkv[2]
            a = F.scaled_dot_product_attention(q, k, v)   # <- Sage patches this
            a = a.transpose(1, 2).reshape(b, s, d)
            x = x + self.proj(a)
            x = x + self.mlp(self.norm2(x))
            return x

    class DiTStack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([DiTBlock() for _ in range(depth)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    return DiTStack()


def _measure(torch: Any, fn, warmup: int, iters: int):
    """Return (median_ms, vram_peak_gb, last_output). Resets the CUDA peak so
    vram_peak reflects this run only."""
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
    vram = torch.cuda.max_memory_allocated() / 1e9
    return statistics.median(samples) * 1000.0, vram, out


def _rel_err(torch: Any, a: Any, b: Any) -> float:
    num = (a.float() - b.float()).norm().item()
    den = b.float().norm().item() or 1.0
    return num / den


def run(config: Optional[Config] = None, seqs=(1024, 4096),
        iters: int = 20, tc_mode: str = "reduce-overhead",
        dim: int = 1536, heads: int = 24, depth: int = 6) -> int:
    if config is None:
        config = load_config()

    print("=" * 72)
    print(" Optimizer V10 — pipeline benchmark (diffusion-shaped; measured)")
    print("=" * 72)
    print("  " + config.profile.summary())

    torch = _probe_torch()
    if torch is None or not (getattr(torch, "cuda", None)
                             and torch.cuda.is_available()):
        print("\n  torch+CUDA not available — refusing to fake an end-to-end "
              "number. Install the acceleration venv and re-run. (0 rows.)")
        return 0
    print(f"  torch {torch.__version__} | device=cuda | "
          f"DiT stack dim={dim} heads={heads} depth={depth} | tc_mode={tc_mode}")

    # backends under test (the runtime swaps activated in Mile 3)
    from v9.accel.registry import build_default_registry
    from v9.accel.torchcompile import TorchCompileBackend
    reg = build_default_registry(config.profile, config)
    sage = reg.get("sage_attention_v2")
    tc = TorchCompileBackend(mode=tc_mode)   # explicit mode for bounded compile

    header = (f"  {'workload':16s} {'backend':16s} {'base_ms':>9s} "
              f"{'accel_ms':>9s} {'speedup':>8s} {'vram_gb':>8s} "
              f"{'rel_err':>8s} {'admit':>6s}")

    dtype = torch.float16
    rows = 0
    for seq in seqs:
        print(f"\n  --- sequence length {seq} (latent tokens) ---")
        print(header)
        print("  " + "-" * (len(header) - 2))

        m = _build_dit_stack(torch, dim, heads, depth).to("cuda", dtype).eval()
        x = torch.randn(1, seq, dim, device="cuda", dtype=dtype)

        base_ms, base_vram, base_out = _measure(
            torch, lambda: m(x), warmup=3, iters=iters)
        print(f"  {('seq'+str(seq)):16s} {'eager(baseline)':16s} "
              f"{base_ms:9.3f} {'—':>9s} {'1.000':>8s} {base_vram:8.2f} "
              f"{'0.000':>8s} {'—':>6s}")

        candidates = []
        if sage is not None and sage.enabled and sage.applies_to(
                "attention", (1, heads, seq, dim // heads), "fp16"):
            candidates.append(("sage_attention_v2",
                               lambda: sage.run_patched(lambda: m(x)),
                               sage.version, 3))
        if tc.enabled and tc.applies_to("dit", (1, seq, dim), "fp16"):
            cm = tc.load(None).runner.wrap(m)        # compiled module
            candidates.append((f"torch_compile[{tc_mode}]",
                               lambda: cm(x), tc.version, 8))  # +warmup→compile

        for name, fn, ver, warm in candidates:
            try:
                acc_ms, acc_vram, acc_out = _measure(
                    torch, fn, warmup=warm, iters=iters)
            except Exception as exc:  # noqa: BLE001 - one backend can't sink it
                print(f"  {('seq'+str(seq)):16s} {name:16s} {'FAILED':>9s}  "
                      f"({type(exc).__name__}: {str(exc)[:40]})")
                continue
            speedup = base_ms / acc_ms if acc_ms > 0 else 1.0
            rel = _rel_err(torch, acc_out, base_out)
            admit = (speedup >= quality.MIN_SPEEDUP) and (rel <= _REL_ERR_MAX)

            quality.record_quality(
                ts_utc=_utc_now_iso(), speedup=speedup,
                lpips_delta=0.0, clip_delta=0.0,   # deferred to image path
                app="comfyui", backend_name=name.split("[")[0],
                backend_version=ver, model_sha256=f"dit_d{depth}_h{heads}",
                shape_w=seq, shape_h=dim, batch=1, precision="fp16",
                notes={"harness": "pipeline_dit", "rel_err": rel,
                       "vram_peak_gb": acc_vram, "base_ms": base_ms,
                       "accel_ms": acc_ms, "tc_mode": tc_mode,
                       "perceptual": "lpips/clip deferred to image path"})
            telemetry.record_generation(
                ts_utc=_utc_now_iso(), workflow_hash=f"dit_seq{seq}",
                model_sha256=f"dit_d{depth}_h{heads}", shape_w=seq, shape_h=dim,
                batch=1, sampler="dit", steps=iters, precision="fp16",
                backend_set=[name], wall_seconds=acc_ms / 1000.0,
                samples_per_sec=(1000.0 / acc_ms if acc_ms else 0.0),
                vram_peak_gb=acc_vram)
            rows += 1
            print(f"  {('seq'+str(seq)):16s} {name:16s} {base_ms:9.3f} "
                  f"{acc_ms:9.3f} {speedup:8.3f} {acc_vram:8.2f} {rel:8.4f} "
                  f"{'YES' if admit else 'no':>6s}")

    print(f"\n  wrote {rows} measured row(s) to quality + telemetry ledgers.")
    print(f"  gate (this harness): speedup >= {quality.MIN_SPEEDUP} AND "
          f"rel_err <= {_REL_ERR_MAX}.  Perceptual LPIPS/CLIP: deferred to the "
          f"real image path (Mile 4b), recorded as 0.0 placeholders + rel_err "
          f"in notes — not faked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
