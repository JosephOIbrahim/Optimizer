"""Generate the Optimizer V9 app icon (multi-resolution .ico + a PNG preview).

Pure-Pillow/numpy render — no external art assets. Design: dark teal-charcoal
rounded tile, faint tensor-core dot grid, bold "V9" with an NVIDIA-green -> cyan
gradient fill and a soft green glow. Reads at 16px (bold V9 dominates).

Run:  D:\\OptimizerV9\\venv\\Scripts\\python.exe assets\\make_icon.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ICO = os.path.join(HERE, "v9.ico")
OUT_PNG = os.path.join(HERE, "v9_icon_preview.png")

S = 1024                       # master render size
RADIUS = 190                   # tile corner radius
BG_TOP = (26, 35, 48)          # #1a2330  dark teal
BG_BOT = (9, 12, 17)           # #090c11  near-black
GREEN = (118, 185, 0)          # NVIDIA green #76b900
CYAN = (0, 224, 255)           # #00e0ff


def vgrad(w: int, h: int, top, bot) -> Image.Image:
    t = np.linspace(0.0, 1.0, h).reshape(h, 1, 1)
    arr = (np.array(top).reshape(1, 1, 3) * (1 - t)
           + np.array(bot).reshape(1, 1, 3) * t)
    arr = np.repeat(arr.astype("uint8"), w, axis=1)
    return Image.fromarray(arr, "RGB")


def load_font(px: int) -> ImageFont.FreeTypeFont:
    for name in ("seguibl.ttf", "segoeuib.ttf", "arialbd.ttf", "impact.ttf"):
        path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                            "Fonts", name)
        if os.path.exists(path):
            return ImageFont.truetype(path, px)
    return ImageFont.load_default()


def fit_font(text: str, target_w: float) -> ImageFont.FreeTypeFont:
    base = 200
    f = load_font(base)
    bbox = f.getbbox(text)
    w = bbox[2] - bbox[0]
    scale = target_w / max(w, 1)
    return load_font(int(base * scale))


def main() -> None:
    # --- background tile ------------------------------------------------
    img = vgrad(S, S, BG_TOP, BG_BOT).convert("RGBA")

    # faint tensor-core dot grid
    overlay = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    step, r = 96, 5
    for y in range(step, S, step):
        for x in range(step, S, step):
            od.ellipse([x - r, y - r, x + r, y + r], fill=(118, 185, 0, 22))
    img = Image.alpha_composite(img, overlay)

    # subtle inner border
    border = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(border)
    bd.rounded_rectangle([28, 28, S - 29, S - 29], radius=RADIUS - 20,
                         outline=(118, 185, 0, 70), width=6)
    img = Image.alpha_composite(img, border)

    # --- "V9" wordmark --------------------------------------------------
    text = "V9"
    font = fit_font(text, target_w=0.60 * S)
    tb = font.getbbox(text)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = (S - tw) // 2 - tb[0]
    ty = (S - th) // 2 - tb[1] - 10  # nudge up for optical centering

    # soft green glow
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((tx, ty), text, font=font,
                              fill=(118, 185, 0, 255))
    glow = glow.filter(ImageFilter.GaussianBlur(24))
    img = Image.alpha_composite(img, glow)

    # gradient-filled crisp text (green top -> cyan bottom)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).text((tx, ty), text, font=font, fill=255)
    grad = vgrad(S, S, GREEN, CYAN).convert("RGBA")
    grad.putalpha(mask)
    img = Image.alpha_composite(img, grad)

    # cyan speed underline
    bar = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(bar).rounded_rectangle(
        [S * 0.30, S * 0.80, S * 0.70, S * 0.80 + 26], radius=13,
        fill=(0, 224, 255, 230))
    img = Image.alpha_composite(img, bar)

    # --- round the corners (transparent outside) ------------------------
    corner = Image.new("L", (S, S), 0)
    ImageDraw.Draw(corner).rounded_rectangle([0, 0, S - 1, S - 1],
                                             radius=RADIUS, fill=255)
    img.putalpha(corner)

    # --- export ---------------------------------------------------------
    preview = img.resize((512, 512), Image.LANCZOS)
    preview.save(OUT_PNG)
    master = img.resize((256, 256), Image.LANCZOS)
    master.save(OUT_ICO, format="ICO",
                sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                       (64, 64), (128, 128), (256, 256)])
    print(f"wrote {OUT_ICO}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
