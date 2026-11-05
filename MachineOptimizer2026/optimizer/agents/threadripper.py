"""Threadripper PRO 7965WX agent (24c/48t, 4 CCDs x 6c).

CCD masks (adjacent-SMT-sibling map — verify once with coreinfo):
  CCD0=0xFFF  CCD1=0xFFF000  CCD2=0xFFF000000  CCD3=0xFFF000000000

Affinity is applied to *running* processes and re-asserted by the watchdog on every
cycle / workload switch (single owner = the orchestrator loop, never a competing
scheduled task). In SEQUENTIAL modes the active app gets the whole box; in PARALLEL
the three apps are fenced to non-overlapping CCDs so they stop fighting for cache.
"""
from __future__ import annotations

from typing import List

from ..base import Agent, Context, Proposal, Risk, WorkloadMode
from ..builders import command, guided, prop, reg

_IFEO = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options"
_ALL = "0xFFFFFFFFFFFF"


def _affinity(id_, procs: str, mask: str, priority: str = None) -> str:
    setp = f"; $_.PriorityClass='{priority}'" if priority else ""
    return (f"Get-Process {procs} -EA SilentlyContinue | "
            f"%{{ try{{ $_.ProcessorAffinity=[IntPtr]{mask}{setp} }}catch{{}} }}")


class ThreadripperAgent(Agent):
    name = "threadripper"
    domain = "threadripper"

    def propose(self, ctx: Context) -> List[Proposal]:
        out: List[Proposal] = []

        out.append(prop(
            "cpu_topology", "Verify CCD/SMT topology before trusting affinity masks",
            self.domain,
            why="The CCD masks assume adjacent SMT siblings (CCD0=cpu0-11 ...). Confirm once so a "
                "wrong map doesn't pin apps to half the cores.",
            impact="correctness guardrail for all affinity settings.", confidence=0.85,
            changes=[guided(
                "cpu_topology", "Confirm the 4-CCD adjacent-sibling map (one time)",
                "Run: coreinfo64.exe -accepteula -c -n -l   (adjacent '**' = SMT siblings; NUMA "
                "nodes = 1 => keep NPS1/UMA). Expected: CCD0=0xFFF, CCD1=0xFFF000, "
                "CCD2=0xFFF000000, CCD3=0xFFF000000000. If the map differs, adjust masks in "
                "optimizer/agents/threadripper.py.")]))

        # --- Houdini: SEQUENTIAL = all-but-top-2 + AboveNormal ---
        out.append(prop(
            "houdini_affinity_seq", "Houdini: all cores (reserve 2) + AboveNormal", self.domain,
            why="Give solo Houdini the whole 7965WX minus 2 logicals for OS/telemetry, at raised "
                "priority for sim/cook responsiveness.",
            impact="max sim/PDG throughput in Houdini-only sessions.", confidence=0.7,
            applies_when=[WorkloadMode.HOUDINI_HEAVY],
            changes=[command("houdini_affinity_seq", "Houdini affinity 0x3FFF.. + AboveNormal",
                             _affinity("h", "houdinifx,houdini,houdinicore", "0x3FFFFFFFFFFF", "AboveNormal"),
                             undo_cmd=_affinity("h", "houdinifx,houdini,houdinicore", _ALL, "Normal"),
                             admin=False, risk=Risk.LOW)]))

        # --- PARALLEL: non-overlapping CCD fences ---
        out.append(prop(
            "houdini_affinity_par", "Houdini -> CCD0+CCD1 (parallel fence)", self.domain,
            why="In parallel mode fence Houdini to CCD0+1 so UE and ComfyUI keep their own CCDs.",
            impact="stops cross-app cache thrash in parallel workflows.", confidence=0.75,
            applies_when=[WorkloadMode.PARALLEL],
            changes=[command("houdini_affinity_par", "Houdini affinity CCD0+1 (0xFFFFFF)",
                             _affinity("h", "houdinifx,houdini,houdinicore", "0xFFFFFF", "AboveNormal"),
                             undo_cmd=_affinity("h", "houdinifx,houdini,houdinicore", _ALL, "Normal"),
                             admin=False, risk=Risk.LOW)]))

        out.append(prop(
            "ue5_affinity_par", "UnrealEditor -> CCD2 (parallel fence)", self.domain,
            why="Fence UE + its ShaderCompileWorkers to CCD2 in parallel mode (workers respawn, so "
                "the watchdog re-applies each cycle).",
            impact="isolates UE shader compile from Houdini/ComfyUI.", confidence=0.7,
            applies_when=[WorkloadMode.PARALLEL],
            changes=[command("ue5_affinity_par", "UE affinity CCD2 (0xFFF000000)",
                             _affinity("u", "UnrealEditor,ShaderCompileWorker", "0xFFF000000"),
                             undo_cmd=_affinity("u", "UnrealEditor,ShaderCompileWorker", _ALL),
                             admin=False, risk=Risk.LOW)]))

        out.append(prop(
            "comfyui_affinity_par", "ComfyUI -> CCD3 (parallel fence)", self.domain,
            why="Fence the ComfyUI python.exe to CCD3 in parallel mode.",
            impact="isolates diffusion CPU work from Houdini/UE.", confidence=0.75,
            applies_when=[WorkloadMode.PARALLEL],
            changes=[command(
                "comfyui_affinity_par", "ComfyUI affinity CCD3 (0xFFF000000000)",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "? { $_.CommandLine -match 'ComfyUI|main.py' } | "
                "%{ try{ (Get-Process -Id $_.ProcessId).ProcessorAffinity=[IntPtr]0xFFF000000000 }catch{} }",
                undo_cmd="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "? { $_.CommandLine -match 'ComfyUI|main.py' } | "
                "%{ try{ (Get-Process -Id $_.ProcessId).ProcessorAffinity=[IntPtr]0xFFFFFFFFFFFF }catch{} }",
                admin=False, risk=Risk.LOW)]))

        # --- persistent IFEO priorities ---
        out.append(prop(
            "husk_priority", "husk.exe persistent AboveNormal priority", self.domain,
            why="Batch/farm Karma renders (husk) run without the UI; give them AboveNormal.",
            impact="faster husk renders on a shared box.", confidence=0.8, admin=True,
            changes=[reg("husk_priority", rf"{_IFEO}\husk.exe\PerfOptions",
                         "CpuPriorityClass", 6, desc="husk CpuPriorityClass=6 (AboveNormal)",
                         admin=True, risk=Risk.LOW)]))
        return out
