"""Optimizer V8 — Launcher (Pentagram-aesthetic Tk UI).

Editorial typography, cream + ink + single burnt-sienna accent, hairline
geometry, no rounding, no gradients. Three flat buttons that shell out to
existing .bat entry points; status footer that polls venv / ComfyUI /
dashboard every five seconds.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import tkinter as tk
from tkinter import font as tkfont


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG       = "#F0EBDF"   # cream paper
INK      = "#1A1A1A"   # deep ink
MUTED    = "#7A7363"   # warm taupe
ACCENT   = "#B23A1F"   # burnt sienna
HAIRLINE = INK

WINDOW_W = 480
WINDOW_H = 760
PAD      = 40

# Family fallback chain: prefer editorial sans if installed, degrade to OS.
FONT_FAMILIES = (
    "Helvetica Neue", "Inter", "GT America", "Suisse Int'l",
    "Segoe UI Variable", "Segoe UI", "Arial",
)

ROOT        = Path(__file__).resolve().parent
START_BAT   = ROOT / "start_optimizer.bat"
INSTALL_BAT = ROOT / "install_and_run.bat"
DASH_URL    = "http://127.0.0.1:8090"
COMFY_URL   = "http://127.0.0.1:8188/system_stats"


# ---------------------------------------------------------------------------
# Status probes
# ---------------------------------------------------------------------------
def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        with urlopen(url, timeout=timeout):
            return True
    except (URLError, OSError):
        return False


def _venv_ok() -> bool:
    venv_python = ROOT / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return False
    try:
        rc = subprocess.run(
            [str(venv_python), "--version"],
            capture_output=True, timeout=3,
        ).returncode
        return rc == 0
    except (subprocess.SubprocessError, OSError):
        return False


class StatusProbe:
    """Polls in a background thread; pushes results back via on_update()."""

    def __init__(self, on_update) -> None:
        self.on_update = on_update
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.on_update({
                "venv":      _venv_ok(),
                "comfyui":   _http_ok(COMFY_URL, 1.0),
                "dashboard": _http_ok(DASH_URL, 1.0),
            })
            self._stop.wait(5.0)


# ---------------------------------------------------------------------------
# Flat button (rectangular, hairline, no rounding)
# ---------------------------------------------------------------------------
class FlatButton(tk.Frame):
    """Two styles: 'primary' (ink fill, cream text) or 'outline'."""

    def __init__(self, master, text: str, command, style: str = "outline",
                 arrow: bool = False, fonts: dict | None = None) -> None:
        super().__init__(master, bg=BG, highlightthickness=1,
                         highlightbackground=INK, bd=0)
        self.command = command
        self.style = style
        self._fonts = fonts or {}

        bg, fg = self._colors(False)
        self.configure(bg=bg)

        body = tk.Frame(self, bg=bg)
        body.pack(fill="x", padx=20, pady=18)

        self.label = tk.Label(
            body, text=text, bg=bg, fg=fg, anchor="w",
            font=self._fonts.get("button"),
        )
        if arrow:
            self.label.pack(side="left", fill="x", expand=True)
            self.arrow = tk.Label(
                body, text="→", bg=bg, fg=fg,
                font=self._fonts.get("button"),
            )
            self.arrow.pack(side="right")
        else:
            self.arrow = None
            self.label.pack(fill="x")

        self._body = body
        self._bind_all((self, body, self.label))
        if self.arrow is not None:
            self._bind_all((self.arrow,))

    def _bind_all(self, widgets) -> None:
        for w in widgets:
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)
            w.bind("<Button-1>", self._click)

    def _colors(self, hover: bool) -> tuple[str, str]:
        if self.style == "primary":
            return (ACCENT if hover else INK, BG)
        return (INK, BG) if hover else (BG, INK)

    def _paint(self, hover: bool) -> None:
        bg, fg = self._colors(hover)
        self.configure(bg=bg)
        self._body.configure(bg=bg)
        self.label.configure(bg=bg, fg=fg)
        if self.arrow is not None:
            self.arrow.configure(bg=bg, fg=fg)

    def _enter(self, _evt) -> None:
        self._paint(True)

    def _leave(self, _evt) -> None:
        self._paint(False)

    def _click(self, _evt) -> None:
        try:
            self.command()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"button error: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
class LauncherWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Optimizer V8")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.minsize(WINDOW_W, WINDOW_H)

        self.fonts = self._build_fonts()
        self._build_layout()

        self.probe = StatusProbe(self._on_status_update)
        self.probe.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------
    def _build_fonts(self) -> dict:
        installed = set(tkfont.families())
        family = next((f for f in FONT_FAMILIES if f in installed), "Arial")
        return {
            "index":   tkfont.Font(family=family, size=11, weight="bold"),
            "title":   tkfont.Font(family=family, size=56, weight="bold"),
            "section": tkfont.Font(family=family, size=10, weight="bold"),
            "body":    tkfont.Font(family=family, size=11, weight="normal"),
            "button":  tkfont.Font(family=family, size=12, weight="bold"),
            "status":  tkfont.Font(family=family, size=11, weight="normal"),
        }

    def _spacer(self, parent, h: int) -> None:
        tk.Frame(parent, bg=BG, height=h).pack(fill="x")

    def _hairline(self, parent, width: int | None = None) -> None:
        rule = tk.Canvas(parent, bg=BG, height=1, highlightthickness=0)
        rule.pack(fill="x" if width is None else "none", anchor="w")
        rule.update_idletasks()
        w = width if width is not None else rule.winfo_width()
        rule.configure(width=w)
        rule.create_line(0, 0, w, 0, fill=HAIRLINE)

    # -----------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=PAD, pady=PAD)

        # ---- Header ---------------------------------------------------
        tk.Label(outer, text="01.", bg=BG, fg=ACCENT,
                 font=self.fonts["index"]).pack(anchor="w")

        self._spacer(outer, 6)

        tk.Label(outer, text="OPTIMIZER", bg=BG, fg=INK,
                 font=self.fonts["title"]).pack(anchor="w")
        tk.Label(outer, text="V8", bg=BG, fg=ACCENT,
                 font=self.fonts["title"]).pack(anchor="w")

        self._spacer(outer, 18)
        tk.Frame(outer, bg=HAIRLINE, height=1).pack(fill="x")
        self._spacer(outer, 14)

        tk.Label(outer, text="GPU ACCELERATION ORCHESTRATOR",
                 bg=BG, fg=MUTED,
                 font=self.fonts["section"]).pack(anchor="w")
        self._spacer(outer, 4)
        tk.Label(outer, text="RTX 4090   ·   SM 89   ·   ADA LOVELACE",
                 bg=BG, fg=MUTED,
                 font=self.fonts["body"]).pack(anchor="w")

        # ---- Buttons --------------------------------------------------
        self._spacer(outer, 44)

        FlatButton(outer, "LAUNCH OPTIMIZER", self._on_launch,
                   style="primary", arrow=True,
                   fonts=self.fonts).pack(fill="x")
        self._spacer(outer, 12)
        FlatButton(outer, "INSTALL  /  REBUILD", self._on_install,
                   style="outline", fonts=self.fonts).pack(fill="x")
        self._spacer(outer, 12)
        FlatButton(outer, "OPEN DASHBOARD", self._on_dashboard,
                   style="outline", fonts=self.fonts).pack(fill="x")

        # ---- Status ---------------------------------------------------
        self._spacer(outer, 48)
        tk.Label(outer, text="STATUS", bg=BG, fg=MUTED,
                 font=self.fonts["section"]).pack(anchor="w")

        self._spacer(outer, 4)
        rule = tk.Canvas(outer, bg=BG, height=1, width=60,
                         highlightthickness=0)
        rule.pack(anchor="w")
        rule.create_line(0, 0, 60, 0, fill=HAIRLINE)
        self._spacer(outer, 14)

        self.status_rows = {}
        for key, label in (("venv", "VENV"),
                           ("comfyui", "COMFYUI"),
                           ("dashboard", "DASHBOARD")):
            row = tk.Frame(outer, bg=BG)
            row.pack(fill="x", pady=2)
            dot = tk.Label(row, text="●", bg=BG, fg=MUTED,
                           font=self.fonts["body"], width=2)
            dot.pack(side="left")
            tk.Label(row, text=label, bg=BG, fg=INK,
                     font=self.fonts["status"], width=12,
                     anchor="w").pack(side="left")
            value = tk.Label(row, text="—", bg=BG, fg=MUTED,
                             font=self.fonts["status"], anchor="w")
            value.pack(side="left")
            self.status_rows[key] = (dot, value)

    # -----------------------------------------------------------------
    def _on_launch(self) -> None:
        self._start_bat(START_BAT)

    def _on_install(self) -> None:
        self._start_bat(INSTALL_BAT)

    def _on_dashboard(self) -> None:
        webbrowser.open(DASH_URL)

    def _start_bat(self, path: Path) -> None:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return
        try:
            os.startfile(str(path))
        except OSError as exc:  # pragma: no cover
            print(f"launch failed: {exc}", file=sys.stderr)

    # -----------------------------------------------------------------
    def _on_status_update(self, state: dict) -> None:
        self.root.after(0, lambda: self._apply_status(state))

    def _apply_status(self, state: dict) -> None:
        labels = {True: ("ONLINE", ACCENT), False: ("OFFLINE", MUTED)}
        for key, (dot, value) in self.status_rows.items():
            text, color = labels[bool(state.get(key))]
            dot.configure(fg=color)
            value.configure(text=text, fg=color)

    # -----------------------------------------------------------------
    def _on_close(self) -> None:
        self.probe.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    LauncherWindow().run()


if __name__ == "__main__":
    main()
