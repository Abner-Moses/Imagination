#!/usr/bin/env python3
"""Assemble the 12 unmodified review renders into a labelled 4x3 sheet."""

from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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
        frame = Image.open(episode / "rgb" / "000000.jpg").convert("RGB")
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        pose = (
            (episode / "sensors_clean.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
        )
        headers = (
            (episode / "sensors_clean.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        )
        row = dict(zip(headers, pose))
        with (episode / "navigation_labels.csv").open(newline="", encoding="utf-8") as handle:
            navigation = next(csv.DictReader(handle))
        label = (
            f"#{index + 1:02d}  {float(row['ground_clearance_m']):.2f} m  |  "
            f"{metadata.get('review_focus') or metadata['clutter_style']}  |  "
            f"{navigation['navigation_class'].replace('_', ' ')}  "
            f"(track {100 * float(navigation['track_pixel_fraction']):.0f}%)  |  "
            f"{metadata['lighting']['condition'].replace('_', ' ')}"
        )
        x, y = (index % columns) * tile_w, (index // columns) * (tile_h + label_h)
        sheet.paste(frame, (x, y))
        draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill=(18, 18, 18))
        draw.text((x + 10, y + tile_h + 7), label, fill=(245, 245, 245), font=font)
    output = root / "contact_sheet.png"
    sheet.save(output, optimize=True)
    print(output)


if __name__ == "__main__":
    main()
