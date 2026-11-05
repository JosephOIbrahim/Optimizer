"""hython: list every light, its light-link and shadow-link collections, and count the
distinct shadow-link sets (Karma XPU allows at most 4 — the source of
'shadowlinking limit of 4 hit, lighting results will be incorrect').

    hython harness\\light_links.py E:\\OptimizerV12\\scenes\\ovoid_hero_v002.usd
"""
import sys
from collections import Counter

from pxr import Usd, UsdLux

stage = Usd.Stage.Open(sys.argv[1])
lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI) or p.IsA(UsdLux.BoundableLightBase)
          or p.IsA(UsdLux.NonboundableLightBase) or p.GetTypeName().endswith("Light")]
print(f"{len(lights)} light prims\n")
sets = Counter()


def col(prim, name):
    c = Usd.CollectionAPI(prim, name)
    if not c:
        return None
    inc = [str(x) for x in c.GetIncludesRel().GetTargets()]
    exc = [str(x) for x in c.GetExcludesRel().GetTargets()]
    rule = c.GetIncludeRootAttr().Get()
    return {"includeRoot": rule, "includes": inc, "excludes": exc}


for p in lights:
    ll = col(p, "lightLink")
    sl = col(p, "shadowLink")
    active = p.GetAttribute("visibility").Get() if p.HasAttribute("visibility") else None
    enabled = p.GetAttribute("inputs:intensity").Get() if p.HasAttribute("inputs:intensity") else None
    key = None
    if sl and (sl["excludes"] or not sl["includeRoot"] or sl["includes"] and sl["includes"] != ["/"]):
        key = (tuple(sl["includes"]), tuple(sl["excludes"]), sl["includeRoot"])
        sets[key] += 1
    tag = "SHADOW-LINKED" if key else ""
    print(f"{p.GetPath()}  [{p.GetTypeName()}]  intensity={enabled}  vis={active}  {tag}")
    if ll and (ll["excludes"] or not ll["includeRoot"] or ll["includes"] not in ([], ["/"])):
        print(f"    lightLink  inc={ll['includes']} exc={ll['excludes']} root={ll['includeRoot']}")
    if key:
        print(f"    shadowLink inc={sl['includes']} exc={sl['excludes']} root={sl['includeRoot']}")

print(f"\ndistinct non-default shadow-link sets: {len(sets)}  (XPU limit 4)")
for k, n in sets.items():
    print(f"  x{n}: includes={list(k[0])} excludes={list(k[1])} includeRoot={k[2]}")
