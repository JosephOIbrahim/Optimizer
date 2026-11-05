"""SideFX Houdini 21 agent. Writes reversible key = value lines into houdini.env
(auto-located under Documents\\houdini2*.*), forcing OpenCL/CUDA onto the 4090,
relocating temp to the fast NVMe, and tuning Karma XPU. In-app-only settings
(OptiX denoiser, PDG slots) are guided."""
from __future__ import annotations

from pathlib import Path
from typing import List

from ..base import Agent, Context, Proposal, Risk, WorkloadMode
from ..builders import file_line, guided, prop

_SEP = " = "


def _houdini_env() -> str:
    docs = Path.home() / "Documents"
    cands = sorted(docs.glob("houdini2*"), reverse=True)  # newest version dir first
    base = cands[0] if cands else (docs / "houdini21.0")
    return str(base / "houdini.env")


class HoudiniAgent(Agent):
    name = "houdini"
    domain = "houdini"

    def propose(self, ctx: Context) -> List[Proposal]:
        envf = _houdini_env()
        nvme = ctx.fastest_nvme or "F:"
        temp = f"{nvme}/houdini_temp".replace("\\", "/")

        def fl(key, name, value, risk=Risk.LOW):
            return file_line(key, envf, name, value, sep=_SEP, risk=risk)

        out: List[Proposal] = []

        out.append(prop(
            "houdini_ocl_nvidia", "Force Houdini OpenCL/CUDA onto the RTX 4090", self.domain,
            why="Pin OpenCL to the NVIDIA device and CUDA to GPU 0 so solvers/XPU never pick a "
                "software or wrong device.",
            impact="reliable GPU solves (Vellum/Pyro/OpenCL SOPs) + Karma XPU on the 4090.",
            confidence=0.9,
            changes=[fl("houdini_ocl_vendor", "HOUDINI_OCL_VENDOR", '"NVIDIA Corporation"'),
                     fl("houdini_ocl_devicenumber", "HOUDINI_OCL_DEVICENUMBER", 0),
                     fl("houdini_ocl_devicetype", "HOUDINI_OCL_DEVICETYPE", "GPU"),
                     fl("houdini_cuda_visible", "CUDA_VISIBLE_DEVICES", 0)]))

        out.append(prop(
            "houdini_temp_nvme", "Houdini temp -> fast NVMe", self.domain,
            why="VDB/sim scratch on the fast empty NVMe instead of the OS drive.",
            impact="TOP WIN: faster VDB/sim caching and .hip autosaves.", confidence=0.95,
            changes=[fl("houdini_temp_dir", "HOUDINI_TEMP_DIR", f'"{temp}"')]))

        out.append(prop(
            "houdini_maxthreads", "Reserve 2 cores for the OS (HOUDINI_MAXTHREADS=-2)",
            self.domain,
            why="Leave 2 logicals for OS + the optimizer's telemetry so a full-throttle solve "
                "doesn't starve the watchdog. Set 24 for SMT-unfriendly sim-heavy sessions.",
            impact="steadier interactivity under heavy solves.", confidence=0.85,
            changes=[fl("houdini_maxthreads", "HOUDINI_MAXTHREADS", -2)]))

        out.append(prop(
            "karma_xpu_tuning", "Karma XPU: multithread AOV/deep blending", self.domain,
            why="Multithread per-device blending for AOV/deep/cryptomatte-heavy XPU renders.",
            impact="faster XPU renders with many AOVs.", confidence=0.8,
            changes=[fl("karma_xpu_blending", "KARMA_XPU_NUM_PER_DEVICE_BLENDING_THREADS", 4)]))

        out.append(prop(
            "houdini_ocl_pool", "OpenCL VRAM pre-alloc pool (solo sim sessions)", self.domain,
            why="Pre-allocate a larger OpenCL pool for solo GPU-sim-heavy sessions; kept modest so "
                "it doesn't collide with UE/ComfyUI in parallel.",
            impact="fewer reallocation stalls in big Vellum/Pyro solves.", confidence=0.7,
            applies_when=[WorkloadMode.HOUDINI_HEAVY, WorkloadMode.IDLE],
            changes=[fl("houdini_ocl_pool", "HOUDINI_OCL_MEMORY_POOL_SIZE", 0.25, risk=Risk.MEDIUM)]))

        # --- in-app (guided) ---
        out.append(prop(
            "karma_optix_denoiser", "Enable NVIDIA OptiX denoiser in Karma", self.domain,
            why="OptiX AI denoiser gives clean frames at far lower sample counts on the 4090.",
            impact="TOP WIN: fewer samples for the same look in Karma XPU.", confidence=0.85,
            changes=[guided(
                "karma_optix_denoiser", "Karma Render Settings > Denoiser = NVIDIA OptiX",
                "On the Karma Render Settings LOP set Denoiser = 'NVIDIA OptiX' "
                "(USD karma:global:denoiser = optix). In the Render View, Denoiser dropdown > "
                "NVIDIA OptiX. Revert: Denoiser = None.")]))

        out.append(prop(
            "pdg_scheduler_slots", "PDG Local Scheduler slots = physical cores", self.domain,
            why="Match TOPs concurrency to 24 physical cores for light items; drop to 4-8 when each "
                "item is a full sim/render to avoid oversubscription.",
            impact="better PDG/TOPs throughput without thrashing.", confidence=0.85,
            changes=[guided(
                "pdg_scheduler_slots", "Local Scheduler > Scheduling > Total Slots = 24 (or 4-8)",
                "On the TOPs Local Scheduler node > Scheduling tab: Total Slots = 24 for light work "
                "items; 4-8 for heavy per-item sims/renders. Default is cores/4. Per-node parm — "
                "reset restores default.")]))
        return out
