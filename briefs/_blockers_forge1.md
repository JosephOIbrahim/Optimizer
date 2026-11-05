# Forge-1 Blockers / Environmental Notes

## Status
**Forge-1 deliverables D1-D4 are complete.** The two items below are environmental blockers I worked around without compromising the deliverables. Surfacing them per Commandment 3 / 4 so the next agent in line is not surprised.

---

## B1. The committed `venv/` is broken

**What:** `C:\Users\User\OneDrive\Desktop\Optimizer_V7_G3\venv\pyvenv.cfg` points to a Python home that does not exist on the dev host:

```
home = C:\Users\User\miniconda3
executable = C:\Users\User\miniconda3\python.exe
version = 3.13.9
```

`C:\Users\User\miniconda3` is missing from disk. Any `venv\Scripts\python.exe` invocation either falls through to system Python (when launched via `activate.bat` in cmd.exe) or fails outright (errno 103 / "did not find executable") when invoked directly.

**What I tried (3 attempts, then escalated):**
1. Direct invocation `venv\Scripts\python.exe -V` -- exit 103, "did not find executable at miniconda".
2. Invocation via `activate.bat` then `python -V` -- silently delegates to system Python 3.14.
3. Copy `python.exe` out of the venv to `/tmp` and run -- exit 106, "failed to locate pyvenv.cfg".

**Workaround used:** All Forge-1 verification was run against system Python 3.14 at `C:\Python314\python.exe`, which has `torch 2.9.1+cu130`, `sageattention`, `onnx 1.21.0`, `numpy 2.2.6`, and `pytest 8.4.2`. 20/22 tests pass; the 2 skipped require TensorRT (see B2).

**Recommended fix (operator decision -- I did not freelance into this):**
- Either reinstall miniconda 3.13 at `C:\Users\User\miniconda3` to restore the venv, or
- Recreate the venv against an available interpreter:
  ```
  rmdir /S /Q venv
  py -3.12 -m venv venv
  venv\Scripts\activate.bat
  pip install -r requirements.txt
  ```

This blocks Forge-2/Forge-3/Forge-4 from running their suites in the documented venv until resolved.

---

## B2. TensorRT 11.x not installed

**What:** `tensorrt` Python package is not installed on system Python 3.14 (where I verified) and could not be installed into the broken venv (B1). The brief flagged this as a likely blocker and instructed `pip install tensorrt --index-url https://pypi.nvidia.com`.

**What I did:** Did NOT attempt the install, because:
1. The venv is broken (B1), so any install would land in system Python, polluting global state without operator consent.
2. TRT is a heavy CUDA dependency; auto-installing it would violate Commandment 5 (role isolation -- environment provisioning is not Forge-1's role).
3. My code already degrades cleanly when TRT is absent; D1-D4 are complete and tested without it.

**Effect on deliverables:** None. `TensorRTBackend.__init__` lazy-imports `tensorrt`, sets `enabled=False` on ImportError, and `compile()` raises with a clear install hint. The two TRT-gated tests (`test_tensorrt_backend_plan_passthrough`, `test_tensorrt_backend_compile_64x64x64_onnx_gemm`) are written and will pass automatically once TRT is installed.

**Recommended fix:** After resolving B1, run from the rebuilt venv:
```
pip install tensorrt --index-url https://pypi.nvidia.com
```
Then re-run `python -m pytest tests/forge1/ -v` -- expect 22/22 passing. If the install fails 3x, do NOT downgrade to TRT 10.x; raise a real blocker for the operator (per the brief).

---

## Not blockers (notes only)

- **CUTLASS 4.x not used in Phase A.** Per brief, `CutlassFP8GemmBackend` is shipped with `enabled=False` and uses the existing 3.5.1 install path via `CUTLASSManager`. No action needed.
- **`_TrtRunner` exposes the IExecutionContext directly.** Forge-3 will need to marshal torch tensors via `tensor.data_ptr()` and call `set_input_shape` + `execute_async_v3` on a torch CUDA stream. Documented in the handoff under D2.
- **Multi-input ONNX models in TRT compile.** Current `compile()` applies the same `ShapeRange` to every network input. Multi-input variant deferred; documented in handoff "Open questions".

None of these block downstream work.
