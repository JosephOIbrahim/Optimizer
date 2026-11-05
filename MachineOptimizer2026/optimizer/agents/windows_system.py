"""Windows 11 system agent. Owns the single active power plan (Ultimate Performance)
and its sub-tweaks, the reboot-required registry set, Defender exclusions, and the
shared scratch/cache directories on the fast NVMe. Pagefile / MPO / per-exe GPU
preference are surfaced as guided steps (large or conditional side effects)."""
from __future__ import annotations

from typing import List

from ..base import Agent, Context, Proposal, Risk
from ..builders import command, defender, guided, prop, reg

_GD = r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
_SMK = r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\kernel"
_SMP = r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power"
_MM = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile"

_ULTIMATE_APPLY = (
    "$u = powercfg /list | Select-String 'Ultimate Performance';"
    "if ($u) { $g = ([regex]'[0-9a-fA-F-]{36}').Match($u.ToString()).Value }"
    "else { $o = powercfg -duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61;"
    " $g = ([regex]'[0-9a-fA-F-]{36}').Match(($o -join ' ')).Value };"
    "powercfg /setactive $g")
_ULTIMATE_UNDO = (
    "$m=([regex]'[0-9a-fA-F-]{36}').Match('{prior}').Value;"
    "if(-not $m){$m='381b4222-f694-41f0-9685-ff5bb260df2e'};"
    "powercfg /setactive $m")


def _pc(setting: str, value, default) -> tuple:
    """Return (apply, undo, snapshot) powercfg commands. undo restores the CAPTURED prior
    AC index ({prior}); the hardcoded default is only a fallback if the snapshot was empty."""
    snap = ("$q = (powercfg /query SCHEME_CURRENT " + setting +
            " | Select-String 'Current AC Power Setting Index' | Select-Object -First 1).Line;"
            " if($q){ [Convert]::ToInt32(($q -replace '.*0x',''),16) } else { '' }")
    a = f"powercfg /setacvalueindex SCHEME_CURRENT {setting} {value}; powercfg /setactive SCHEME_CURRENT"
    u = (f"$v='{{prior}}'; if($v -eq ''){{$v='{default}'}};"
         f" powercfg /setacvalueindex SCHEME_CURRENT {setting} $v; powercfg /setactive SCHEME_CURRENT")
    return a, u, snap


class WindowsSystemAgent(Agent):
    name = "windows_system"
    domain = "windows_system"

    def propose(self, ctx: Context) -> List[Proposal]:
        nvme = ctx.fastest_nvme or "F:"
        out: List[Proposal] = []

        # --- scratch/cache dirs (needed by houdini/unreal/nvidia/comfyui) ---
        dirs = f"'{nvme}\\houdini_temp','{nvme}\\UnrealDDC','{nvme}\\OptimizerV9\\cuda_jit_cache','D:\\OptimizerV9\\cuda_cache'"
        out.append(prop(
            "scratch_dirs", "Create scratch/cache dirs on fast NVMe", self.domain,
            why="Houdini temp, UE DDC and CUDA caches live on the fast empty NVMe, off the OS drive.",
            impact="faster scratch I/O; prerequisite for the NVMe path settings.", confidence=0.9,
            changes=[command("scratch_dirs", f"mkdir {nvme}\\houdini_temp, {nvme}\\UnrealDDC, cuda caches",
                             f"New-Item -ItemType Directory -Force -Path {dirs} | Out-Null",
                             undo_cmd=f"foreach($p in @({dirs})){{ if((Test-Path $p) -and -not (Get-ChildItem $p)){{ Remove-Item $p -Force }} }}",
                             admin=False, risk=Risk.NONE)]))

        # --- power plan (single owner) + sub-tweaks on SCHEME_CURRENT ---
        out.append(prop(
            "power_plan_ultimate", "Activate Ultimate Performance power plan", self.domain,
            why="Unparks cores and removes idle/boost hesitation that starves the GPU feed.",
            impact="steadier CPU clocks for sims + shader compile.", confidence=0.95, admin=True,
            changes=[command("power_plan_ultimate", "Ultimate Performance plan active",
                             _ULTIMATE_APPLY, undo_cmd=_ULTIMATE_UNDO,
                             snapshot_cmd="powercfg /getactivescheme",
                             verify_cmd="if((powercfg /getactivescheme) -match 'Ultimate'){exit 0}else{exit 1}",
                             admin=True, risk=Risk.LOW, power_plan=True)]))

        a, u, s = _pc("SUB_PROCESSOR PROCTHROTTLEMAX", 100, 100)
        out.append(prop(
            "cpu_max_state_100", "Processor max performance state = 100%", self.domain,
            why="Guarantees full boost headroom (MAX=100 is unambiguously safe; MIN left default "
                "to preserve single-core boost residency for Houdini VEX cooks).",
            impact="removes CPU max-state throttling.", confidence=0.78, admin=True,
            changes=[command("cpu_max_state_100", "PROCTHROTTLEMAX=100", a, undo_cmd=u,
                             snapshot_cmd=s, admin=True, risk=Risk.LOW, power_plan=True)]))

        a, u, s = _pc("SUB_PROCESSOR 0cc5b647-c1df-4637-891a-dec35c318583", 100, 0)
        out.append(prop(
            "core_parking_off", "Disable CPU core parking", self.domain,
            why="Keeps all 48 threads unparked so sims/compiles don't wait on core wake-up.",
            impact="lower scheduling latency on the 7965WX.", confidence=0.9, admin=True,
            changes=[command("core_parking_off", "min unparked cores = 100%", a, undo_cmd=u,
                             snapshot_cmd=s, admin=True, risk=Risk.LOW, power_plan=True)]))

        a, u, s = _pc("SUB_PCIEXPRESS ee12f906-d277-404b-b6da-e5fa1a576df5", 0, 2)
        out.append(prop(
            "pcie_aspm_off", "PCIe Link State Power Management = Off", self.domain,
            why="Stops the x16 Gen4 link idling down (it currently drops to Gen1 at idle).",
            impact="keeps full PCIe bandwidth to the 4090 under load.", confidence=0.82, admin=True,
            changes=[command("pcie_aspm_off", "ASPM index = 0 (Off)", a, undo_cmd=u,
                             snapshot_cmd=s, admin=True, risk=Risk.LOW, power_plan=True)]))

        # --- registry ---
        out.append(prop(
            "game_mode_off", "Windows Game Mode OFF", self.domain,
            why="Game Mode's scheduler heuristics can hurt long-running pro app throughput.",
            impact="removes a scheduler wildcard for VFX apps.", confidence=0.6,
            changes=[reg("game_mode_off", r"HKCU\Software\Microsoft\GameBar",
                         "AutoGameModeEnabled", 0, desc="AutoGameModeEnabled=0", admin=False,
                         risk=Risk.NONE)]))

        out.append(prop(
            "hags_on", "Hardware-Accelerated GPU Scheduling ON (reboot)", self.domain,
            why="Currently OFF (HwSchMode=1) on this box. HAGS improves GPU submission latency "
                "and is a prerequisite for modern present paths.",
            impact="TOP WIN: lower GPU dispatch latency after reboot.", confidence=0.9, admin=True,
            changes=[reg("hags_on", _GD, "HwSchMode", 2, desc="HwSchMode=2 (reboot)",
                         admin=True, risk=Risk.LOW)]))

        out.append(prop(
            "gpu_msi_mode", "GPU MSI (Message-Signaled Interrupts) ON (reboot)", self.domain,
            why="MSI mode reduces interrupt latency vs line-based IRQs for the GPU.",
            impact="smoother GPU interrupt handling.", confidence=0.85, admin=True,
            changes=[command(
                "gpu_msi_mode", "MSISupported=1 for the RTX 4090 (reboot)",
                "$id=(Get-PnpDevice -Class Display | ? FriendlyName -like '*NVIDIA*').InstanceId;"
                "$k=\"HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\$id\\Device Parameters\\Interrupt Management\\MessageSignaledInterruptProperties\";"
                "New-Item $k -Force | Out-Null; Set-ItemProperty $k -Name MSISupported -Type DWord -Value 1",
                snapshot_cmd="$id=(Get-PnpDevice -Class Display | ? FriendlyName -like '*NVIDIA*').InstanceId;"
                "(Get-ItemProperty \"HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\$id\\Device Parameters\\Interrupt Management\\MessageSignaledInterruptProperties\" -Name MSISupported -EA SilentlyContinue).MSISupported",
                undo_cmd="$id=(Get-PnpDevice -Class Display | ? FriendlyName -like '*NVIDIA*').InstanceId;"
                "$k=\"HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\$id\\Device Parameters\\Interrupt Management\\MessageSignaledInterruptProperties\";"
                "$v='{prior}'; if($v -eq ''){ Remove-ItemProperty $k -Name MSISupported -EA SilentlyContinue }"
                " else { Set-ItemProperty $k -Name MSISupported -Type DWord -Value ([int]$v) }",
                admin=True, risk=Risk.MEDIUM)]))

        for key, name, val, path, undo_val, desc in [
            ("timer_resolution", "GlobalTimerResolutionRequests", 1, _SMK, None,
             "GlobalTimerResolutionRequests=1 (reboot)"),
            ("mmcss_responsiveness", "SystemResponsiveness", 10, _MM, 20,
             "SystemResponsiveness=10 (reboot)"),
            ("network_throttling_off", "NetworkThrottlingIndex", 4294967295, _MM, 10,
             "NetworkThrottlingIndex=off (reboot)"),
            ("fast_startup_off", "HiberbootEnabled", 0, _SMP, 1, "HiberbootEnabled=0"),
        ]:
            out.append(prop(
                key, desc.split("=")[0].replace("_", " ").title(), self.domain,
                why="Latency/consistency tuning for sustained pro workloads.",
                impact="marginal but safe; reversible.", confidence=0.72, admin=True,
                changes=[reg(key, path, name, val, desc=desc, admin=True, risk=Risk.LOW)]))

        # --- Defender exclusions (huge win for file-churn workloads) ---
        # Scope to specific cache/scratch dirs only — never the whole drive root (the NVMe
        # could be probed as any fixed volume), which would weaken AV coverage too broadly.
        excl = [f"{nvme}\\houdini_temp", f"{nvme}\\UnrealDDC", f"{nvme}\\OptimizerV9",
                "D:\\OptimizerV9", "D:\\OptimizerV8_EngineCache"]
        out.append(prop(
            "defender_exclusions", "Defender exclusions for scratch/cache + render procs",
            self.domain,
            why="Real-time AV scanning of Houdini/UE/ComfyUI cache churn is a large hidden tax.",
            impact="TOP WIN: big I/O speedup for VDB/DDC/engine-cache reads/writes.",
            confidence=0.95, admin=True,
            changes=[defender(f"defender_path_{i}", p) for i, p in enumerate(excl)] + [
                command("defender_procs", "exclude render processes",
                        "Add-MpPreference -ExclusionProcess 'houdini.exe','husk.exe','hython.exe','UnrealEditor.exe'",
                        snapshot_cmd="((Get-MpPreference).ExclusionProcess) -join ','",
                        undo_cmd="$prior='{prior}'; foreach($x in 'houdini.exe','husk.exe','hython.exe','UnrealEditor.exe')"
                                 "{ if($prior -notmatch [regex]::Escape($x)){ Remove-MpPreference -ExclusionProcess $x -EA SilentlyContinue } }",
                        admin=True, risk=Risk.LOW)]))

        out.append(prop(
            "memory_compression_off", "Disable memory compression (128GB RAM)", self.domain,
            why="With 128GB RAM, compressing the standby list wastes CPU for no benefit.",
            impact="frees CPU cycles during heavy paging of large scenes.", confidence=0.65,
            admin=True,
            changes=[command("memory_compression_off", "Disable-MMAgent -MemoryCompression (reboot)",
                             "Disable-MMAgent -MemoryCompression",
                             undo_cmd="Enable-MMAgent -MemoryCompression", admin=True, risk=Risk.LOW)]))

        # --- guided (large or conditional side effects) ---
        out.append(prop(
            "pagefile_nvme", "Fixed pagefile on fast NVMe (guided)", self.domain,
            why="A dedicated fixed pagefile on the fast NVMe helps large-scene paging, but it "
                "consumes up to 128GB of scratch space — your call, not auto-applied.",
            impact="steadier paging under 128GB+ working sets.", confidence=0.85,
            changes=[guided(
                "pagefile_nvme", "Set C: system-managed + fixed pagefile on the NVMe",
                f"Admin PowerShell:\n"
                f"Set-CimInstance (Get-CimInstance Win32_ComputerSystem) -Property @{{AutomaticManagedPageFile=$false}}\n"
                f"Set-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management' "
                f"-Name PagingFiles -Type MultiString -Value @('C:\\pagefile.sys 0 0','{nvme}\\pagefile.sys 32768 131072')\n"
                f"Reboot. Revert: set AutomaticManagedPageFile=$true, reboot, delete {nvme}\\pagefile.sys.")]))

        out.append(prop(
            "mpo_conditional", "Disable MPO ONLY if XDR flickers (guided)", self.domain,
            why="Multi-Plane Overlay is left ON by default; only disable if you see XDR flicker.",
            impact="conditional fix for compositor flicker.", confidence=0.7,
            changes=[guided(
                "mpo_conditional", "If (and only if) XDR flickers: DisableOverlays=1",
                f'reg add "{_GD}" /v DisableOverlays /t REG_DWORD /d 1 /f   (reboot). '
                "Revert: reg delete the value, reboot. Confirm via dxdiag > Display > MPO.")]))
        return out
