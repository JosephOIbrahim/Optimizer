"""NVIDIA driver / NVAPI-profile agent.

Auto-applies the two safe env-var levers (CUDA JIT cache size + path). Everything
that lives in the driver profile database is surfaced as a precise GUIDED step,
because the classic NVIDIA Control Panel was retired 2026-05-26 — those settings now
live in the NVIDIA App / NVIDIA Profile Inspector and cannot be safely poked via a
registry write. Global sysmem-fallback is deliberately LEFT at Driver Default; only
the per-app 'No Sysmem Fallback' profiles are recommended (synthesis decision).
"""
from __future__ import annotations

from typing import List

from ..base import Agent, Context, Proposal, Risk, Scope
from ..builders import env, guided, prop

_INSPECTOR = ("NVIDIA App > Graphics, or NVIDIA Profile Inspector for scripted/backup. "
              "BACK UP FIRST in Profile Inspector: toolbar 'Export user defined profiles' "
              "-> D:\\OptimizerV9\\nvidia_profiles_backup.nip.")


class NvidiaDriverAgent(Agent):
    name = "nvidia_driver"
    domain = "nvidia_driver"

    def propose(self, ctx: Context) -> List[Proposal]:
        nvme = ctx.fastest_nvme or "F:"
        jit = f"{nvme}\\OptimizerV9\\cuda_jit_cache"
        out: List[Proposal] = []

        # --- auto-applied env vars (no admin) ---
        out.append(prop(
            "cuda_cache_maxsize", "CUDA JIT cache = 4 GiB (max)", self.domain,
            why="Larger CUDA compile cache => fewer JIT recompiles for Karma XPU / ComfyUI kernels.",
            impact="-shader/kernel recompile stalls across CUDA apps.", confidence=0.9,
            changes=[env("cuda_cache_maxsize", "CUDA_CACHE_MAXSIZE", 4294967296,
                         desc="CUDA_CACHE_MAXSIZE=4 GiB", risk=Risk.NONE)]))

        out.append(prop(
            "cuda_cache_path", "CUDA JIT cache -> fastest NVMe", self.domain,
            why=f"Keep the CUDA compute cache on the fast scratch NVMe ({nvme}), off the OS drive.",
            impact="faster cold-cache kernel loads; complements D:\\OptimizerV9 caches.",
            confidence=0.75,
            changes=[env("cuda_cache_path", "CUDA_CACHE_PATH", jit,
                         desc=f"CUDA_CACHE_PATH={jit}", risk=Risk.NONE)],
            note="dir auto-created by windows_system scratch-dir setup"))

        # --- guided: driver-profile settings (NVIDIA App / Profile Inspector) ---
        out.append(prop(
            "sysmem_fallback_per_app", "CUDA Sysmem Fallback = OFF (per-app)", self.domain,
            why="On a 24GB card with 65GB shared sysmem, silent VRAM->RAM spill turns an OOM "
                "into a 5-20x slowdown. Per-app 'No Fallback' keeps Karma/ComfyUI VRAM-resident "
                "and fails fast. GLOBAL stays Driver Default (near-24GB husk/full-precision jobs "
                "may legitimately prefer spilling over crashing).",
            impact="TOP WIN: eliminates silent PCIe paging in Houdini/Karma XPU + ComfyUI.",
            confidence=0.85, admin=False,
            changes=[guided(
                "sysmem_fallback_per_app",
                "Set 'CUDA - Sysmem Fallback Policy' = 'Prefer No Sysmem Fallback' per-app",
                f"{_INSPECTOR}\n"
                "NVIDIA App > Graphics > Program Settings > Add each of:\n"
                "  - houdini.exe  and  husk.exe   (Houdini 21 bin)\n"
                "  - UnrealEditor.exe\n"
                "  - D:\\OptimizerV9\\venv\\Scripts\\python.exe   (ComfyUI interpreter)\n"
                "For each set 'CUDA - Sysmem Fallback Policy' = 'Prefer No Sysmem Fallback' > Apply.\n"
                "Leave GLOBAL / Base Profile at 'Driver Default'. Revert: set entry to 'Driver "
                "Default' or reimport the .nip backup.")]))

        out.append(prop(
            "power_mgmt_max_perf", "Power mgmt = Prefer Max Performance (per-app)", self.domain,
            why="Removes clock ramp-up latency for the viewport/render apps; GLOBAL stays Optimal.",
            impact="steadier clocks during interactive Houdini/UE work.", confidence=0.88,
            changes=[guided(
                "power_mgmt_max_perf",
                "Set 'Power management mode' = 'Prefer maximum performance' per-app",
                "NVIDIA App > Graphics > Program Settings > houdini.exe, husk.exe, UnrealEditor.exe: "
                "'Power management mode' = 'Prefer maximum performance' > Apply. Leave GLOBAL at "
                "'Optimal power'. Revert: set back to 'Normal/Optimal power'.")]))

        out.append(prop(
            "cuda_force_p2_off", "CUDA Force P2 State = Off", self.domain,
            why="Runs GDDR6X at full P0 memory clock for CUDA compute instead of the P2 downclock.",
            impact="higher memory bandwidth for Karma XPU / ComfyUI (small heat cost).",
            confidence=0.85, admin=True,
            changes=[guided(
                "cuda_force_p2_off", "Profile Inspector (as admin): CUDA - Force P2 State = Off",
                f"{_INSPECTOR}\nRun NVIDIA Profile Inspector AS ADMIN > Base Profile > 'Common' > "
                "'CUDA - Force P2 State' = Off (0x00000000) > Apply. Effect on next CUDA context; "
                "no reboot. Revert: set = On (driver default) or reimport the .nip.")]))

        out.append(prop(
            "shader_cache_unlimited", "Driver Shader Cache Size = Unlimited", self.domain,
            why="Avoids eviction of the driver's DXCache/GLCache shader disk cache.",
            impact="fewer shader recompiles across sessions.", confidence=0.85,
            changes=[guided(
                "shader_cache_unlimited", "NVIDIA App > Global > Shader Cache Size = Unlimited",
                "NVIDIA App > Graphics > Global Settings > 'Shader Cache Size' = 'Unlimited' (or "
                "'100 GB') > Apply. Revert: 'Driver Default'. Cache: %LOCALAPPDATA%\\NVIDIA\\DXCache.")]))

        out.append(prop(
            "viewport_quality_profiles", "Viewport driver tweaks (texture filter / threaded opt)",
            self.domain,
            why="High-performance texture filtering + threaded GL optimization for the viewport apps.",
            impact="minor viewport responsiveness; A/B for Houdini GL.", confidence=0.6,
            changes=[guided(
                "viewport_quality_profiles", "Per-app texture filtering + threaded optimization",
                "NVIDIA App > Program Settings > houdini.exe & UnrealEditor.exe: 'Texture filtering "
                "- Quality' = 'High performance'. houdini.exe: 'Threaded optimization' = 'On' "
                "(A/B vs Auto). Revert: back to 'Quality' / 'Auto'.")]))

        # --- guardrail: verify the color-critical XDR was never degraded ---
        out.append(prop(
            "xdr_color_guardrail", "GUARDRAIL: keep XDR at 10/12-bit + RGB + Full + HDR",
            self.domain,
            why="The Apple Pro Display XDR is a color-critical reference display. No optimization "
                "may downgrade its output. This is a check-and-preserve, never a change.",
            impact="protects color accuracy for review/grading.", confidence=0.8,
            changes=[guided(
                "xdr_color_guardrail", "Verify (never change) XDR color output",
                "AFTER applying everything, confirm in NVIDIA App > Display: Output color depth = "
                "10/12 bpc, format = RGB, dynamic range = Full; Windows Settings > System > Display "
                "> HDR = On. If any regressed (e.g. after a driver event), restore them. The "
                "watchdog also flags this.")]))
        return out
