#!/usr/bin/env python3
"""Download and checksum the small CC0 Poly Haven asset bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "third_party" / "polyhaven"
MODEL_IDS = [
    "dirty_football",
    "rock_07",
    "rock_09",
    "stone_01",
    "namaqualand_stones_01",
    "cardboard_box_01",
    "trashbag",
    "plastic_monobloc_chair_01",
]
TEXTURE_IDS = ["grass_ground", "withered_grass", "brown_mud"]


def download(url: str, path: Path, expected_md5: str):
    # Poly Haven publishes MD5 values for file integrity, not authentication.
    if path.is_file() and hashlib.md5(path.read_bytes()).hexdigest() == expected_md5:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    payload = response.content
    actual = hashlib.md5(payload).hexdigest()
    if actual != expected_md5:
        raise RuntimeError(f"Checksum mismatch for {url}: {actual} != {expected_md5}")
    path.write_bytes(payload)
    print(f"downloaded {path.relative_to(ROOT)} ({len(payload) / 1_000_000:.2f} MB)")


def api_json(endpoint: str) -> dict:
    response = requests.get(f"https://api.polyhaven.com/{endpoint}", timeout=60)
    response.raise_for_status()
    return response.json()


def download_model(asset_id: str) -> dict:
    files = api_json(f"files/{asset_id}")
    record = files["blend"]["1k"]["blend"]
    target = DEST / asset_id
    download(record["url"], target / f"{asset_id}_1k.blend", record["md5"])
    for relative, included in record.get("include", {}).items():
        download(included["url"], target / relative, included["md5"])
    return api_json(f"info/{asset_id}")


def download_texture(asset_id: str) -> dict:
    files = api_json(f"files/{asset_id}")
    target = DEST / asset_id / "textures"
    choices = {
        "diff": files["Diffuse"]["1k"]["jpg"],
        "nor_gl": files["nor_gl"]["1k"]["jpg"],
        "rough": files["Rough"]["1k"]["jpg"],
    }
    for suffix, record in choices.items():
        download(record["url"], target / f"{asset_id}_{suffix}_1k.jpg", record["md5"])
    return api_json(f"info/{asset_id}")


def main():
    metadata = {"license": "CC0 1.0", "license_url": "https://polyhaven.com/license", "assets": {}}
    for asset_id in MODEL_IDS:
        metadata["assets"][asset_id] = download_model(asset_id)
    for asset_id in TEXTURE_IDS:
        metadata["assets"][asset_id] = download_texture(asset_id)
    (DEST / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"asset bundle ready: {DEST}")


if __name__ == "__main__":
    main()
