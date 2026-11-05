"""Unreal Engine 5.6+ agent. Auto-applies the two machine-level, project-agnostic
levers (DerivedDataCache on the fast NVMe + a longer GPU TDR timeout for offline
renders). All the rendering cvars are project-scoped, so they are delivered as a
ready-to-paste MachineOptimizer_UE56.ini snippet rather than auto-edited into an
unknown project's DefaultEngine.ini."""
from __future__ import annotations

from typing import List

from ..base import Agent, Context, Proposal, Risk, Scope
from ..builders import env, guided, prop, reg

_GD = r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"


class UnrealAgent(Agent):
    name = "unreal"
    domain = "unreal"

    def propose(self, ctx: Context) -> List[Proposal]:
        nvme = ctx.fastest_nvme or "F:"
        ddc = f"{nvme}\\UnrealDDC"
        out: List[Proposal] = []

        out.append(prop(
            "ue_ddc_nvme", "UE DerivedDataCache -> fast NVMe", self.domain,
            why="A large local DDC on the fast NVMe slashes cold shader/asset build times; keeps "
                "the OS drive free.",
            impact="TOP WIN: much faster cook/shader iteration after first build.", confidence=0.88,
            changes=[env("ue_ddc_nvme", "UE-LocalDataCachePath", ddc, scope=Scope.USER,
                         desc=f"UE-LocalDataCachePath={ddc}", risk=Risk.LOW)],
            note="restart Epic Launcher + Editor after applying"))

        out.append(prop(
            "ue_tdr_timeout", "Raise Windows TDR timeout for long GPU renders (reboot)",
            self.domain,
            why="Default ~2s TDR kills long path-traced / Lumen builds. 60s lets them finish; this "
                "only lengthens TDR, never disables it, so it can't risk boot.",
            impact="offline/path-traced renders no longer trip the driver reset.", confidence=0.85,
            admin=True,
            changes=[reg("tdr_delay", _GD, "TdrDelay", 60, desc="TdrDelay=60s (reboot)",
                         admin=True, risk=Risk.MEDIUM),
                     reg("tdr_ddi_delay", _GD, "TdrDdiDelay", 60, desc="TdrDdiDelay=60s (reboot)",
                         admin=True, risk=Risk.MEDIUM)]))

        out.append(prop(
            "ue_engine_ini", "Apply the UE5.6 rendering profile (guided .ini)", self.domain,
            why="Lumen HWRT, sized Nanite/texture/VSM pools for 24GB, DLSS Ray Reconstruction, "
                "OptiX path-tracer denoise and full shader-compiler thread use — project-scoped, "
                "so delivered as a snippet you paste, not an auto-edit of an unknown project.",
            impact="TOP WINS: HW ray-traced Lumen + right-sized VRAM pools + fast shader compile.",
            confidence=0.85,
            changes=[guided(
                "ue_engine_ini", "Paste MachineOptimizer_UE56.ini into your project's DefaultEngine.ini",
                "1) Open MachineOptimizer_UE56.ini (in this optimizer folder). Back up your "
                "<Project>\\Config\\DefaultEngine.ini, then append the [SystemSettings], "
                "[/Script/Engine.RendererSettings] and [DevOptions.Shaders] blocks.\n"
                "2) Project Settings > Rendering: enable 'Support Hardware Ray Tracing' (SM6).\n"
                "3) Install the NVIDIA DLSS/Streamline plugin matching your UE minor version, "
                "enable it (SR/DLAA/RR + 2x Frame Gen on Ada).\n"
                "4) Editor Preferences > Performance: UNCHECK 'Override game screen percentage "
                "settings with editor settings in PIE' (allows DLAA/DLSS in-viewport).\n"
                "Revert: restore the DefaultEngine.ini backup and disable the plugin.")]))
        return out
