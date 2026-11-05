"""
The Jev invariant, enforced by structure rather than by review.

    A Jev judgment is NEVER evidence. Nothing it outputs may reach
    Gate.evaluate, a Verdict, a Baseline, Conditions, an Objective, the
    control-flow fields of an Observation, a Change, or an actuator.

A code reviewer can miss one import. These tests cannot:

  * import closure — in a FRESH interpreter, importing the gate, the actuators,
    the ABI, the baseline, the objective, truth and the plane must leave
    `v12.jev` and `typesafe_sdk` out of sys.modules. If any of them ever imports
    jev, even indirectly through a helper, this fails. (When plane gains display
    annotations it may import jev only lazily, inside the annotation function;
    this test still holds, because module import is what it checks.)
  * a fresh interpreter, because this test process has already imported v12.jev
    through tests/test_jev.py — sys.modules here proves nothing.
  * source scan — no file in v12/ other than jev.py may even NAME typesafe_sdk,
    and jev.py must not import it at module level.
  * positive controls — jev.py exists, does name the SDK, and importing it (and
    asking available()/status_line()) still does not load the SDK. Without
    these the closure test would pass vacuously on a tree with no jev at all.

The subprocess gets an explicit environment with V12_JEV=on and a dummy key, so
the switch being on is shown not to pull anything in. Nothing here touches the
network: no code path in these imports constructs a client.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "v12"

#: Everything that decides, measures or mutates. None of these may reach jev.
GATE_MODULES = (
    "v12",
    "v12.admission",
    "v12.change",
    "v12.baseline",
    "v12.objective",
    "v12.truth",
    "v12.plane",
    "v12.ledger",
    "v12.guard",
    "v12.power",
    "v12.actuators",
)

#: Modules whose source may not reference jev at all, lazily or otherwise:
#: the ones that produce or consume evidence. plane is not listed — Step 4 may
#: give it a lazy display hook — and is held by the closure test instead.
NO_JEV_SOURCE = ("admission.py", "change.py", "baseline.py", "objective.py",
                 "truth.py", "ledger.py", "guard.py", "power.py", "actuators")


def _fresh(code: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() not in (
        "TYPESAFE_API_KEY", "V12_JEV", "PYTHONSTARTUP")}
    env.update({"V12_JEV": "on", "TYPESAFE_API_KEY": "dummy-not-a-real-key",
                "PYTHONDONTWRITEBYTECODE": "1"})
    # Not -I: isolated mode drops the user site, which is where typesafe_sdk is
    # installed on this box, and an SDK that cannot load proves nothing.
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


_PRELUDE = (
    "import sys, json, importlib, pkgutil; sys.path.insert(0, %r)\n" % str(ROOT)
)


class TestImportClosure(unittest.TestCase):
    def test_the_gate_and_its_neighbours_never_load_jev_or_the_sdk(self) -> None:
        code = _PRELUDE + (
            "mods = %r\n"
            "for m in mods: importlib.import_module(m)\n"
            "import v12.actuators as a\n"
            "subs = [n.name for n in pkgutil.iter_modules(a.__path__, 'v12.actuators.')]\n"
            "for m in subs: importlib.import_module(m)\n"
            "print(json.dumps({'jev': 'v12.jev' in sys.modules,"
            " 'sdk': any(k == 'typesafe_sdk' or k.startswith('typesafe_sdk.')"
            " for k in sys.modules), 'subs': subs,"
            " 'loaded': sorted(k for k in sys.modules if k.startswith('v12'))}))\n"
        ) % (GATE_MODULES,)
        r = _fresh(code)
        self.assertTrue(r["subs"], "no actuator submodules found — the walk is broken")
        self.assertIn("v12.plane", r["loaded"])
        self.assertFalse(r["jev"], "importing the gate's modules loaded v12.jev")
        self.assertFalse(r["sdk"], "importing the gate's modules loaded typesafe_sdk")

    def test_importing_jev_and_asking_its_status_does_not_load_the_sdk(self) -> None:
        code = _PRELUDE + (
            "import importlib.util, v12.jev as j\n"
            "ok, why = j.available(); line = j.status_line()\n"
            "print(json.dumps({'jev': 'v12.jev' in sys.modules,"
            " 'sdk': any(k.startswith('typesafe_sdk') for k in sys.modules),"
            " 'installed': importlib.util.find_spec('typesafe_sdk') is not None,"
            " 'line': line}))\n"
        )
        r = _fresh(code)
        if not r["installed"]:
            self.skipTest("typesafe_sdk is not installed for this interpreter, so "
                          "'not loaded' would prove nothing")
        self.assertEqual(r["line"], "jev: on (model jev-1.13.0)")
        self.assertTrue(r["jev"])
        self.assertFalse(r["sdk"], "v12.jev imports typesafe_sdk at module level")
        self.assertTrue(r["line"].startswith("jev: "))

    def test_the_cli_loads_jev_only_when_it_prints_the_jev_line(self) -> None:
        code = _PRELUDE + (
            "import v12.cli\n"
            "print(json.dumps({'jev': 'v12.jev' in sys.modules,"
            " 'sdk': any(k.startswith('typesafe_sdk') for k in sys.modules)}))\n"
        )
        r = _fresh(code)
        self.assertFalse(r["jev"])
        self.assertFalse(r["sdk"])


def _py_files():
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node, a.name
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            yield node, mod
            for a in node.names:
                yield node, f"{mod}.{a.name}" if mod else a.name


class TestSourceScan(unittest.TestCase):
    def test_jev_exists_and_is_the_only_module_that_names_the_sdk(self) -> None:
        jev = PKG / "jev.py"
        self.assertTrue(jev.is_file(), "v12/jev.py is missing")
        self.assertIn("typesafe_sdk", jev.read_text(encoding="utf-8"))
        offenders = [str(p.relative_to(ROOT)) for p in _py_files()
                     if p != jev and "typesafe_sdk" in p.read_text(encoding="utf-8")]
        self.assertEqual(offenders, [])

    def test_jev_never_imports_the_sdk_at_module_level(self) -> None:
        tree = ast.parse((PKG / "jev.py").read_text(encoding="utf-8"))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [name for n in top for _, name in _imports(n)]
        self.assertFalse([n for n in names if "typesafe" in n], names)

    def test_evidence_modules_never_reference_jev(self) -> None:
        offenders = []
        for p in _py_files():
            rel = p.relative_to(PKG)
            if rel.parts[0] not in NO_JEV_SOURCE:
                continue
            for node, name in _imports(ast.parse(p.read_text(encoding="utf-8"))):
                if name.split(".")[-1] == "jev" or ".jev" in name or name == "jev":
                    offenders.append(f"{rel}:{node.lineno} {name}")
        self.assertEqual(offenders, [])

    def test_plane_has_no_module_level_jev_import(self) -> None:
        tree = ast.parse((PKG / "plane.py").read_text(encoding="utf-8"))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [name for n in top for _, name in _imports(n)]
        self.assertFalse([n for n in names if n.endswith("jev") or ".jev" in n], names)

    def test_plane_imports_jev_only_inside_the_annotation_helper(self) -> None:
        """Step 4 gave plane its one lazy display hook, annotate_processes. Any
        other function importing jev is a second path for a label to travel."""
        tree = ast.parse((PKG / "plane.py").read_text(encoding="utf-8"))
        owners = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node, name in _imports(fn):
                if name.endswith("jev") or ".jev" in name:
                    owners.append(fn.name)
        self.assertEqual(sorted(set(owners)), ["annotate_processes"])


if __name__ == "__main__":
    unittest.main()
