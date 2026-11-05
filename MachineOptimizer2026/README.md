# MachineOptimizer 2026

A machine-specific optimizer for **this exact workstation** — RTX 4090 · Threadripper PRO 7965WX · 128 GB · Apple Pro Display XDR (6K) · Windows 11 Pro for Workstations — tuned for **NVIDIA products and VFX** as of **July 2026**.

Not a generic tweaker. Every setting was researched and adversarially verified against this box's reality: consumer **GeForce under WDDM** (so half the "nvidia-smi power-limit" advice is inapplicable and gets rejected), **driver 610.62 / CUDA 13.3**, a **color-critical 6K XDR** that must never be degraded, and the **24 GB VRAM + 65 GB shared sysmem** surface where silent fallback is the #1 hidden perf-killer.

---

## Launch

```
Optimize-This-Machine.bat        # self-elevates, previews, then starts the watchdog
Restore-This-Machine.bat         # reverts everything to the machine's original state
```

The launcher previews the full plan (changes nothing), then — on your confirmation — starts the **continuous watchdog**: it applies the optimizations, writes a restore point first, and keeps re-optimizing as you switch between Houdini, Unreal, and ComfyUI.

Manual / CI use:

```
python -m optimizer.orchestrator --dry-run            # preview only (default)
python -m optimizer.orchestrator --apply --yes        # apply once, no prompt
python -m optimizer.orchestrator --watch              # continuous watchdog
python -m optimizer.orchestrator --restore            # undo everything
python -m optimizer.orchestrator --status             # live machine readout
python -m optimizer.orchestrator --workload houdini_heavy --dry-run   # preview a profile
```

---

## Architecture

```
Optimize-This-Machine.bat
        │  self-elevate → venv → run
        ▼
  orchestrator ──► probe (live GPU/CPU/VRAM/thermal/workload/NVMe)
        │
        ├─► agent team  (each proposes reversible Changes with confidence)
        │      nvidia_driver · windows_system · threadripper
        │      houdini · unreal · comfyui · thermal_power
        │
        ├─► resolve conflicts + dedupe + VRAM budget
        │
        ├─► executor  (the ONLY thing that mutates state)
        │      env_var · registry · power_plan · defender · file_line · command · guided
        │
        └─► safety  (snapshot originals → master restore point → verify → rollback)
```

**One idea makes it all reversible:** agents never touch the machine. They emit declarative `Change` objects, and the executor snapshots each target's *original* value before applying it. The snapshot is recorded once (even as the watchdog re-applies across cycles), so `--restore` always returns the box to its pre-optimizer state.

### Dynamic workflows
The watchdog detects the active workload and re-plans + re-fences VRAM on every switch:

| Workload | Trigger | VRAM posture (24 GB) |
|----------|---------|----------------------|
| `idle` | no VFX app | baseline, compositor only |
| `houdini_heavy` | Houdini/Karma/husk | Houdini gets the card |
| `ue5_heavy` | UnrealEditor | UE5 gets the card |
| `comfyui_heavy` | python + ComfyUI | ComfyUI gets the card |
| `parallel` | ≥2 apps | fenced split + reserve |

### Agent team
| Agent | Owns |
|-------|------|
| `nvidia_driver` | CUDA sysmem-fallback policy, power mode, HAGS, shader cache, XDR 10-bit/HDR (kept, never downgraded) |
| `windows_system` | Ultimate Performance plan, core parking, Defender scratch exclusions, timer, Game Mode |
| `threadripper` | per-app CPU affinity / priority across the 24c/48t, 4-CCD topology |
| `houdini` | Houdini 21 env, Karma XPU/OptiX, PDG cook count, temp on fast NVMe |
| `unreal` | UE5.6 Engine.ini — Lumen HWRT, Nanite pool, VSM, DLSS/RR, DDC on NVMe |
| `comfyui` | expandable_segments, SageAttention, Nunchaku INT4, TRT — wired into the live `D:\OptimizerV9` caches |
| `thermal_power` | honest guardrails: monitor temp/power/throttle, back off — no fake power-limit writes |

---

## Safety model

- **Dry-run is the default.** Nothing changes until you `--apply` or start the watchdog.
- **Every change is snapshotted and reversible.** `Restore-This-Machine.bat` returns the machine to its original state.
- **Admin-gated.** Registry and Defender changes only run elevated; a non-admin run applies only env/config changes and reports the rest.
- **XDR-protected.** No color/bit-depth/HDR downgrade on the reference display, ever.
- **Thermally aware.** The watchdog holds new changes at ≥87 °C and flags active throttling.
- **Honest about limits.** Anything that can't be scripted safely on consumer GeForce/WDDM (undervolt, fan curve, power limit) is surfaced as a *guided* manual step, not faked.

---

## Relationship to `D:\OptimizerV9`

This tool tunes the **system, driver, and applications**. Your live `D:\OptimizerV9` (torchinductor / CUTLASS / TensorRT engine caches, the V11 roofline/autotune loop) is the **inference-acceleration engine** — a different layer. The `comfyui` agent *points at and warms* those caches rather than duplicating them.

---

## Files

```
Optimize-This-Machine.bat      Restore-This-Machine.bat
config.toml                    requirements.txt   (only 'rich', optional)
optimizer/
  orchestrator.py   probe.py   workflows.py   executor.py   safety.py
  base.py           ui.py      hardware_profile.py
  agents/           nvidia_driver · windows_system · threadripper
                    houdini · unreal · comfyui · thermal_power
data/               restore_points/   transactions.jsonl   (created at runtime)
```

Runs on the Python standard library (uses `nvidia-smi`, PowerShell, `winreg`). `rich` only prettifies output.
