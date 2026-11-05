"""hython: split one RenderSettings into a BEAUTY pass and a DATA pass (two
RenderSettings prims, each with its own product file), the standard VFX multipass
layout: beauty.exr carries beauty + lighting components + light groups + denoise;
data.exr carries Cryptomatte + P/N/motion/albedo at a low sample count.

    hython harness\\multipass_usd.py --usd hero.usd --settings /Render/OVOID_CPU_RENDER_V013 \
        --out E:\\OptimizerV12\\scenes\\ovoid_hero_v002_multipass.usd --data-samples 16

Prints the two settings paths to pass to husk -s. Beauty pixel samples are left
as authored; data samples are set on the DATA prim via karma:global:pathtracedsamples.
"""
import argparse
from pathlib import Path

from pxr import Sdf, Usd, UsdRender

DATA_EXACT = {"hitpz", "hitn", "motionvectors", "albedo", "worldp", "depth", "p", "n", "z", "normal", "position", "velocity", "motion"}
DATA_PREFIX = ("crypto",)

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--settings", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data-samples", type=int, default=16)
a = ap.parse_args()

stage = Usd.Stage.Open(a.usd)
src = stage.GetPrimAtPath(a.settings)
rs = UsdRender.Settings(src)
if not rs:
    raise SystemExit(f"no RenderSettings at {a.settings}")

beauty_path = Sdf.Path(a.settings + "_BEAUTY")
data_path = Sdf.Path(a.settings + "_DATA")
# Flatten, then copy the settings prim spec (with its child Products) twice and
# retarget each copy's products rel to its own children.
root = stage.Flatten()
old = Sdf.Path(a.settings)
for dst in (beauty_path, data_path):
    Sdf.CopySpec(root, old, root, dst)
stage2 = Usd.Stage.Open(root)
for dst in (beauty_path, data_path):
    rel = UsdRender.Settings(stage2.GetPrimAtPath(dst)).GetProductsRel()
    rel.SetTargets([t.ReplacePrefix(old, dst) if t.HasPrefix(old) else t for t in rel.GetTargets()])

def split(settings_path, keep_data: bool, samples: int = None):
    rs2 = UsdRender.Settings(stage2.GetPrimAtPath(settings_path))
    for ppath in rs2.GetProductsRel().GetTargets():
        prod = UsdRender.Product(stage2.GetPrimAtPath(ppath))
        rel = prod.GetOrderedVarsRel()
        allv = [str(t) for t in rel.GetTargets()]
        def is_data(t):
            n = t.rsplit("/", 1)[-1].lower()
            return n in DATA_EXACT or n.startswith(DATA_PREFIX)
        keep = [t for t in allv if is_data(t) == keep_data]
        rel.SetTargets(keep)
        name = prod.GetProductNameAttr().Get() or ""
        tag = "data" if keep_data else "beauty"
        if name:
            p = Path(name)
            prod.GetProductNameAttr().Set(str(p.with_name(f"{p.stem}_{tag}{p.suffix or '.exr'}")))
        print(f"{settings_path}: {len(allv)} -> {len(keep)} vars ({tag}): {[k.rsplit('/',1)[-1] for k in keep]}")
    if samples:
        prim = stage2.GetPrimAtPath(settings_path)
        attr = prim.CreateAttribute("karma:global:pathtracedsamples", Sdf.ValueTypeNames.Int)
        attr.Set(samples)
        print(f"{settings_path}: pathtracedsamples = {samples}")

split(beauty_path, keep_data=False)
split(data_path, keep_data=True, samples=a.data_samples)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
root.Export(str(out))
print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
print(f"husk -s {beauty_path}   /   husk -s {data_path} -p {a.data_samples}")
