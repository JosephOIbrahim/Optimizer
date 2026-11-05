"""hython: report RenderSettings resolution/camera and every asset path that does not resolve.
    hython harness\\check_usd.py E:\\OptimizerV12\\scenes\\ovoid_hero_v002.usd
"""
import os
import sys

from pxr import Sdf, Usd, UsdRender

path = sys.argv[1]
stage = Usd.Stage.Open(path)
print("stage:", path)
for prim in stage.Traverse():
    if prim.IsA(UsdRender.Settings):
        rs = UsdRender.Settings(prim)
        res = rs.GetResolutionAttr().Get()
        cams = rs.GetCameraRel().GetTargets()
        print(f"RenderSettings {prim.GetPath()}  res={res}  camera={[str(c) for c in cams]}")
missing = []
checked = 0
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        tn = attr.GetTypeName()
        if tn not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
            continue
        vals = attr.Get()
        if vals is None:
            continue
        for v in (vals if tn == Sdf.ValueTypeNames.AssetArray else [vals]):
            p = v.resolvedPath or v.path
            if not p:
                continue
            checked += 1
            if "<UDIM>" in p or "%(UDIM)d" in p:
                continue
            if not os.path.exists(p):
                missing.append((str(attr.GetPath()), v.path, p))
print(f"asset paths checked: {checked}  missing: {len(missing)}")
for a, raw, p in missing[:40]:
    print("  MISSING", a, "->", raw, "| resolved:", p)
