"""
A guard against scripts with side effects living in the test directory.

What happened
-------------
On 2026-11-05 an agent wrote a scratch script at
`tests/test_metric_cli_refusal.py` to drive `optimize.main()` end to end.
The script had module-level code rather than code inside a test function, so
pytest COLLECTED it and EXECUTED it on import.

`optimize.main()` with the `aov_variant` lever renders the hero scene for
real. So the file rendered the scene — and because it sat in `tests/`, it
fired during two OTHER agents' `pytest tests/ -q` runs, not just its author's.
The author did not see it happen. Eighteen render frames were left in the
measurement store with no measurement record behind them.

The cost was not the eighteen frames. It was that a run in one agent's
directory executed work in another agent's run, invisibly.

Why a test catches it and a convention does not
----------------------------------------------
The convention is "put tests in tests/". The mistake did not break the
convention — it USED it, because the script had to be in `tests/` for pytest
to pick it up. What was missing was a check on what a collected test file
does at import time.

So this test parses every `tests/test_*.py` with `ast` and fails if any
module-level statement calls something that renders or measures. Module-level
code is fine for definitions, imports, constants and `if __name__` guards —
it is the CALLS that are dangerous, because pytest runs them at collection.

Nothing here renders, touches the GPU, or writes outside a temp dir.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

#: Calls that render or measure. These are the ones that cost GPU time and
#: leave files behind. A module-level call to any of them is a side effect
#: waiting to happen.
DANGEROUS_CALLS = {
    "render_shot",
    "render_frames_separately",
    "render_frame",
    "husk",
    "main",
}

#: Modules whose `main` is safe to call at import time.
SAFE_MAIN_MODULES = {"unittest", "pytest"}


def _module_level_calls(tree: ast.AST) -> list:
    """Every call made at MODULE level — outside any function or class.

    A call inside a `def` is a test body and runs only when pytest runs it. A
    call inside a `class` body runs at import (a method definition does not,
    but a class attribute assignment does). This walks the module body and
    descends into class bodies, because those DO execute at import.
    """
    calls = []

    def visit_body(body, in_class=False):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue                      # the body runs later, not now
            if isinstance(node, ast.ClassDef):
                # The class body itself executes at import.
                visit_body(node.body, in_class=True)
                continue
            if isinstance(node, ast.If):
                # `if __name__ == "__main__":` is the standard guard for
                # exactly this, and pytest does NOT set __name__ to
                # "__main__" when it imports a test module. A module-level
                # `if` whose test is not the __main__ guard still executes.
                test = node.test
                is_main_guard = (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                )
                if not is_main_guard:
                    visit_body(node.body, in_class)
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                calls.append(node.value)
                continue
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                calls.append(node.value)
                continue
            if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
                calls.append(node.value)
                continue
            # A bare call statement, and a call wrapped in anything else
            # (a comparison, a subscript, an f-string, a raise) is still a
            # call. The incident shape was `optimize.main()` as a statement,
            # and walking only Expr/Assign would have missed it.
            if isinstance(node, ast.Call):
                calls.append(node)
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    calls.append(child)
                    break

    visit_body(tree.body)
    return calls


def _call_name(node: ast.Call) -> str:
    """The name being called, as written."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _call_module(node: ast.Call) -> str:
    """The module a call is qualified by, if any."""
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.value.id
    return ""


class TestNoSideEffectsAtImport(unittest.TestCase):
    """A collected test file must not render when it is imported."""

    def _test_files(self):
        return sorted(TESTS.glob("test_*.py"))

    def test_the_guard_would_see_this_file(self):
        """The guard must actually be checking something.

        A guard that finds no files is a guard that finds nothing, forever.
        """
        files = self._test_files()
        self.assertGreater(len(files), 5,
                           "the guard should be scanning the real test files")

    def test_no_test_file_renders_at_import(self):
        offenders = []
        for path in self._test_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"),
                                 filename=str(path))
            except SyntaxError:
                continue
            for call in _module_level_calls(tree):
                name = _call_name(call)
                if name not in DANGEROUS_CALLS:
                    continue
                module = _call_module(call)
                if name == "main" and module in SAFE_MAIN_MODULES:
                    continue
                offenders.append(f"{path.relative_to(REPO)}: {name}()")
        self.assertEqual(offenders, [],
                         "these calls would run when pytest IMPORTS the file, "
                         "not when it runs a test: " + "; ".join(offenders))

    def test_the_guard_detects_a_side_effect(self):
        """The guard has to actually catch one, or it is decoration.

        This builds the shape that caused the incident — module-level code in
        a file named test_*.py — and checks that the guard sees it.
        """
        offender = (
            "import harness.shot as shot\n"
            "def helper():\n"
            "    return 1\n"
            "shot.render_shot('x.usd', 1, 503)   # module level: runs on import\n"
        )
        tree = ast.parse(offender)
        names = {_call_name(c) for c in _module_level_calls(tree)}
        self.assertIn("render_shot", names,
                      "the guard must see a module-level render call")

    def test_the_guard_catches_the_exact_incident_shape(self):
        """The incident was a bare `optimize.main()` at module level.

        An earlier version of this guard walked only `Expr` and `Assign`
        statements and MISSED it, because `optimize.main()` as a statement is
        an `Expr` — but only if the walker descends past it. Every shape of a
        module-level call has to be caught, or the guard catches the shapes
        nobody writes.
        """
        shapes = {
            "bare statement": "import optimize\noptimize.main()\n",
            "behind an assignment": "rc = optimize.main()\n",
            "inside a class body": "class T:\n    rc = optimize.main()\n",
        }
        for label, src in shapes.items():
            with self.subTest(shape=label):
                names = {_call_name(c) for c in _module_level_calls(ast.parse(src))}
                self.assertIn("main", names,
                              f"the guard must catch a module-level call: {label}")

    def test_a_class_body_call_is_caught(self):
        """A class body executes at import time; only its METHODS do not."""
        src = "class T:\n    rc = optimize.main()\n"
        names = {_call_name(c) for c in _module_level_calls(ast.parse(src))}
        self.assertIn("main", names)

    def test_a_call_inside_a_function_is_not_an_offender(self):
        """The normal shape — a render inside a test body — is fine."""
        ok = (
            "import harness.shot as shot\n"
            "def test_something():\n"
            "    shot.render_shot('x.usd', 1, 503)\n"
        )
        tree = ast.parse(ok)
        names = {_call_name(c) for c in _module_level_calls(tree)}
        self.assertNotIn("render_shot", names,
                         "a call inside a function is a test body, not an "
                         "import-time side effect")

    def test_a_main_guard_is_not_an_offender(self):
        """`if __name__ == "__main__":` is the standard way to write a
        script, and pytest does not run it on import."""
        guarded = (
            "def main():\n"
            "    print('hi')\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        tree = ast.parse(guarded)
        names = {_call_name(c) for c in _module_level_calls(tree)}
        self.assertNotIn("main", names,
                         "a __main__ guard must not be treated as an "
                         "import-time side effect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
