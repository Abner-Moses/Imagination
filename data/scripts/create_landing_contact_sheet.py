#!/usr/bin/env python3
"""Create a color-overlay review of the per-pixel landing-loss targets."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = np.asarray(((220, 35, 35), (245, 175, 35), (35, 190, 75)), dtype=np.uint8)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "visual_review").resolve()
    episodes = sorted(root.glob("episode_*"))
    if len(episodes) != 12:
        raise SystemExit(f"Expected 12 visual-review episodes, found {len(episodes)}")
    tile_w, tile_h, label_h, columns = 640, 480, 34, 4
    sheet = Image.new("RGB", (columns * tile_w, 3 * (tile_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, episode in enumerate(episodes):
        rgb = np.asarray(
            Image.open(episode / "rgb" / "000000.jpg").convert("RGB"), dtype=np.float32
        )
        mask = np.asarray(
            Image.open(episode / "landing_suitability" / "000000.png"), dtype=np.uint8
        )
        overlay = (0.56 * rgb + 0.44 * COLORS[np.clip(mask, 0, 2)]).astype(np.uint8)
        with (episode / "navigation_labels.csv").open(newline="", encoding="utf-8") as handle:
            navigation = next(csv.DictReader(handle))
        label = (
            f"#{index + 1:02d}  red unsafe / amber caution / green safe  |  "
            f"{navigation['navigation_class'].replace('_', ' ')}  |  risk {float(navigation['risk_score']):.2f}"
        )
        x, y = (index % columns) * tile_w, (index // columns) * (tile_h + label_h)
        sheet.paste(Image.fromarray(overlay, mode="RGB"), (x, y))
        draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill=(18, 18, 18))
        draw.text((x + 10, y + tile_h + 7), label, fill=(245, 245, 245), font=font)
    output = root / "landing_contact_sheet.png"
    sheet.save(output, optimize=True)
    print(output)


if __name__ == "__main__":
    main()
