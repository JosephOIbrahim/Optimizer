"""
comfy_run — time ONE fixed-seed ComfyUI workflow N times. The diffusion objective.

    E:\\OptimizerV12\\venv\\Scripts\\python.exe comfy_run.py --workflow my_api_workflow.json --reps 6

The workflow must be the API-format JSON (ComfyUI: Settings > Dev mode > "Save (API
format)"). Fix the seed inside it and leave it fixed; the harness refuses to
randomize anything because a different seed is a different image and the
guardrails would be comparing noise.

Primary:    frame_ms   queue-to-finished wall time, lower is better
Guardrails: lpips      LPIPS(alex) distance of each image to the REFERENCE image
                       (lower is better; reference = rep 1 of the baseline run,
                       or --reference for a candidate run)
            clip       cosine similarity of CLIP image embeddings to the reference
                       (higher is better). Definition chosen here; V9's was not
                       recorded, so this is the one V12 gates on from now on.
ComfyUI must already be running (default http://127.0.0.1:8188). The harness
does not launch it: how it was launched IS the thing being measured.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import common, throttle  # noqa: E402


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def _post_json(url: str, payload: dict):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run_once(server: str, workflow: dict, poll_s: float = 0.25):
    t0 = time.perf_counter()
    pid = _post_json(f"{server}/prompt", {"prompt": workflow})["prompt_id"]
    while True:
        hist = json.loads(_get(f"{server}/history/{pid}"))
        if pid in hist and hist[pid].get("status", {}).get("completed", False) or (
                pid in hist and hist[pid].get("outputs")):
            break
        time.sleep(poll_s)
    t = time.perf_counter() - t0
    outputs = hist[pid]["outputs"]
    img = None
    for node in outputs.values():
        for im in node.get("images", []):
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                        "type": im.get("type", "output")})
            img = _get(f"{server}/view?{q}")
            break
        if img:
            break
    return t * 1000.0, img


class Guardrails:
    """LPIPS + CLIP similarity to a reference. Loaded lazily; missing packages
    degrade to None rather than aborting the timing run."""

    def __init__(self):
        self.ok = False
        try:
            import torch, lpips, open_clip  # noqa: F401
            from PIL import Image  # noqa: F401
            self.torch, self.lpips_mod, self.open_clip, self.Image = torch, lpips, open_clip, Image
            self.lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
            self.clip, _, self.pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            self.clip = self.clip.cuda().eval()
            self.ok = True
        except Exception as e:
            print(f"guardrails unavailable ({type(e).__name__}: {e}); timing only")

    def _img(self, data: bytes):
        return self.Image.open(io.BytesIO(data)).convert("RGB")

    def score(self, ref: bytes, cand: bytes):
        if not self.ok:
            return None, None
        torch = self.torch
        with torch.no_grad():
            a, b = self._img(ref), self._img(cand)
            ta = self.lpips_mod.im2tensor(self._np(a)).cuda()
            tb = self.lpips_mod.im2tensor(self._np(b)).cuda()
            lp = float(self.lp(ta, tb).item())
            ea = self.clip.encode_image(self.pre(a).unsqueeze(0).cuda())
            eb = self.clip.encode_image(self.pre(b).unsqueeze(0).cuda())
            ea, eb = ea / ea.norm(), eb / eb.norm()
            cs = float((ea * eb).sum().item())
        return lp, cs

    @staticmethod
    def _np(img):
        import numpy as np
        return np.asarray(img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", required=True, help="API-format workflow JSON with a FIXED seed")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--reps", type=int, default=6, help="incl. 1 warm-up")
    ap.add_argument("--reference", help="reference PNG from the baseline run (candidate runs)")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()

    wf_path = Path(args.workflow)
    workflow = json.loads(wf_path.read_text(encoding="utf-8"))
    if "nodes" in workflow and "links" in workflow:
        print("this is the UI-format workflow; export the API format (Dev mode > Save (API format))")
        return 2
    try:
        stats = json.loads(_get(f"{args.server}/system_stats"))
    except Exception as e:
        print(f"ComfyUI not reachable at {args.server} ({e}). Launch it first — the launch flags are part of what is measured.")
        return 2
    # V12's preflight counts ComfyUI as a GPU app. It IS the workload here, so the
    # operator is told what else is running but a running ComfyUI alone does not block.
    if not args.skip_preflight:
        common.preflight()

    common.ensure_dirs()
    out_dir = common.MEASURE_ROOT / "comfy_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    guards = Guardrails()
    ref = Path(args.reference).read_bytes() if args.reference else None

    print(f"\nComfyUI {stats.get('system', {}).get('comfyui_version', '?')}  "
          f"torch {stats.get('system', {}).get('pytorch_version', '?')}  "
          f"workflow {wf_path.name}  x{args.reps} (first is warm-up)\n")
    frame_ms, lp_s, clip_s, thr, images = [], [], [], [], []
    for i in range(args.reps):
        a = throttle.Snapshot.take()
        ms, img = run_once(args.server, workflow)
        b = throttle.Snapshot.take()
        d = throttle.delta(a, b)
        print(f"rep {i}: {ms:9.0f} ms  {'warm-up, discarded' if i == 0 else ''}  [{throttle.describe(d)}]")
        if i == 0:
            continue
        frame_ms.append(ms)
        thr.append(d)
        if img:
            p = out_dir / f"{tag}_{args.label}_rep{i:02d}.png"
            p.write_bytes(img)
            images.append(str(p))
            if ref is None:
                ref = img  # baseline: rep 1 is the reference
                ref_path = p
            lp, cs = guards.score(ref, img)
            if lp is not None:
                lp_s.append(lp)
                clip_s.append(cs)

    summ = common.summarize(frame_ms)
    print(f"\nframe_ms: mean {summ['mean']:.0f}  median {summ['median']:.0f}  cv {summ.get('cv', 0):.2%}")
    if lp_s:
        print(f"lpips: {common.series(lp_s, 4)}\nclip:  {common.series(clip_s, 4)}")
    rec = common.write_measurement("diffusion", {
        "objective": "diffusion", "harness": "comfy_run", "label": args.label,
        "workflow": str(wf_path), "server": args.server, "system_stats": stats,
        "frame_ms": frame_ms, "lpips": lp_s, "clip": clip_s, "throttle": thr,
        "images": images, "reference": args.reference or (str(ref_path) if images else None),
        "summary": summ,
    })
    print(f"\nsaved {rec}\n")
    g = {"lpips": lp_s, "clip": clip_s} if lp_s else None
    for line in common.paste_lines("diffusion", frame_ms, "comfy_run", g):
        print(line)
    if images and not args.reference:
        print(f"# candidate runs must pass:  --reference \"{images[0]}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
