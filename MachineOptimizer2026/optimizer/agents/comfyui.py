"""ComfyUI agent. Reuses the live D:\\OptimizerV9 acceleration caches (torch.compile
inductor cache) rather than duplicating them, and delivers a tuned launcher. The
launch-time flags + per-app sysmem-fallback profile live in Launch-ComfyUI-Optimized.bat
(generated in this folder), not in global env, so they don't leak to other Python."""
from __future__ import annotations

from typing import List

from ..base import Agent, Context, Proposal
from ..builders import guided, prop

_PY = r"D:\OptimizerV9\venv\Scripts\python.exe"       # verified: 3.12.10, torch 2.8.0+cu126
_TRT = r"D:\OptimizerV9\engine_cache\comfyui"


class ComfyUIAgent(Agent):
    name = "comfyui"
    domain = "comfyui"

    def propose(self, ctx: Context) -> List[Proposal]:
        out: List[Proposal] = []

        # NOTE: TORCHINDUCTOR_CACHE_DIR / TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER are set
        # PROCESS-scoped inside Launch-ComfyUI-Optimized.bat, not as global user env vars, so
        # they cannot affect other Python on the box. See the launcher.
        out.append(prop(
            "comfyui_launcher", "Tuned ComfyUI launcher (expandable_segments, --fast, Sage)",
            self.domain,
            why="expandable_segments allocator + Ada FP16 accumulation + SageAttention + a 6K-HDR "
                "VRAM reserve, launched on the validated D:\\OptimizerV9 venv. Kept in a launcher "
                "so these don't affect other Python processes.",
            impact="TOP WIN: faster, VRAM-safer diffusion/video inference on the 4090.",
            confidence=0.85,
            changes=[guided(
                "comfyui_launcher", "Use Launch-ComfyUI-Optimized.bat (generated in this folder)",
                "Launch ComfyUI via Launch-ComfyUI-Optimized.bat (in this optimizer folder). It "
                f"runs {_PY} C:\\ComfyUI\\main.py with:\n"
                "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  --disable-cuda-malloc\n"
                "  --fast fp16_accumulation  --use-sage-attention  --reserve-vram 1.5  --cache-ram 8\n"
                "COMFYUI_ONLY adds --highvram; PARALLEL uses --normalvram (pass 'parallel' as arg). "
                "Revert: launch ComfyUI the old way. Also add the ComfyUI python.exe to the per-app "
                "'No Sysmem Fallback' profile (see nvidia_driver).")]))

        out.append(prop(
            "comfyui_nunchaku_int4", "Prefer INT4 Nunchaku (SVDQuant) for Flux on Ada", self.domain,
            why="INT4 SVDQuant beats FP8 on Ada and fits Flux-family in ~6GB, leaving room for "
                "--highvram. The PyPI name 'nunchaku' is an unrelated squatter — use the official "
                "GitHub wheel matching this venv (torch2.8, cp312).",
            impact="TOP WIN: largest models fit in 24GB with headroom.", confidence=0.72,
            changes=[guided(
                "comfyui_nunchaku_int4", "Install the official Nunchaku wheel (NOT pip nunchaku)",
                f'{_PY} -m pip install --no-deps '
                '"https://github.com/nunchaku-tech/nunchaku/releases/download/vX.Y.Z/'
                'nunchaku-<ver>+torch2.8-cp312-cp312-win_amd64.whl"  (get the exact filename from '
                "the Nunchaku GitHub releases page). Then clone nunchaku-tech/ComfyUI-nunchaku into "
                "C:\\ComfyUI\\custom_nodes and use the INT4 loader nodes. Revert: remove the custom "
                "node folder + pip uninstall nunchaku.")]))

        out.append(prop(
            "comfyui_trt_junction", "Consolidate TensorRT engines onto D: (guided)", self.domain,
            why="Junction the ComfyUI TensorRT engine dir onto the existing D:\\OptimizerV9 engine "
                "cache so TRT engines are reused, not rebuilt.",
            impact="reuses prebuilt TRT engines across ComfyUI installs.", confidence=0.5,
            changes=[guided(
                "comfyui_trt_junction", "mklink /J the ComfyUI tensorrt dir -> D:\\OptimizerV9",
                f'After installing the ComfyUI-TensorRT node: '
                f'cmd /c mklink /J "C:\\ComfyUI\\models\\tensorrt" "{_TRT}"  '
                f'(create the target first). Revert: rmdir the junction (leaves the real cache).')]))
        return out
