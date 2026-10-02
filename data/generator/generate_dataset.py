#!/usr/bin/env python3
"""Host-side CLI that resolves config and launches the BlenderProc pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER_APP = Path("/Applications/Blender.app")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "field.yaml"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--trajectories-per-layout", type=int)
    parser.add_argument("--frames-per-trajectory", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument(
        "--blender-app", default=os.environ.get("BLENDER_APP", str(DEFAULT_BLENDER_APP))
    )
    parser.add_argument(
        "--noise", action="store_true", help="Enable configured synthetic sensor noise"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete episodes and regenerate only partial/missing episodes",
    )
    parser.add_argument(
        "--visual-review",
        action="store_true",
        help="Render only the 12-frame approval contact-sheet set",
    )
    parser.add_argument(
        "--visual-approved",
        action="store_true",
        help="Explicit gate required for runs larger than 12 frames",
    )
    return parser.parse_args()


def make_blenderproc_shim(blender_app: Path) -> Path:
    binary = blender_app / "Contents" / "MacOS" / "Blender"
    resources = blender_app / "Contents" / "Resources"
    if not binary.is_file():
        raise FileNotFoundError(f"Blender executable not found: {binary}")
    versions = sorted(
        p for p in resources.iterdir() if p.is_dir() and p.name.replace(".", "").isdigit()
    )
    if not versions:
        raise RuntimeError(f"Cannot locate Blender version resources below {resources}")
    shim = PROJECT_ROOT / ".blenderproc_blender"
    shim.mkdir(exist_ok=True)
    contents_link = shim / "Contents"
    version_link = shim / versions[-1].name
    # Keep both BlenderProc's package cache and every mutable path repository-local.
    # Older runs used a Contents symlink; migrate that safe, generated shim in place.
    if contents_link.is_symlink():
        contents_link.unlink()
    (contents_link / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents_link / "Resources").mkdir(parents=True, exist_ok=True)
    binary_link = contents_link / "MacOS" / "Blender"
    resource_link = contents_link / "Resources" / versions[-1].name
    if not binary_link.exists():
        binary_link.symlink_to(binary)
    if not resource_link.exists():
        resource_link.symlink_to(versions[-1], target_is_directory=True)
    if not version_link.exists():
        version_link.symlink_to(versions[-1], target_is_directory=True)
    return shim


def detect_blender_python_version(blender_app: Path) -> str:
    """Return the executable name of Blender's embedded Python."""
    resources = blender_app / "Contents" / "Resources"
    versions = sorted(
        p for p in resources.iterdir() if p.is_dir() and p.name.replace(".", "").isdigit()
    )
    python_bins = sorted((versions[-1] / "python" / "bin").glob("python3.*"))
    python_bins = [path for path in python_bins if path.is_file()]
    if not python_bins:
        raise RuntimeError("Could not detect Blender's bundled Python executable")
    python_name = python_bins[-1].name
    return python_name


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    overrides = {
        "num_episodes": args.num_episodes,
        "trajectories_per_layout": args.trajectories_per_layout,
        "frames_per_trajectory": args.frames_per_trajectory,
        "base_seed": args.base_seed,
    }
    for key, value in overrides.items():
        if value is not None:
            config["dataset"][key] = value
    if args.noise:
        config["sensors"]["noise_enabled"] = True
    if args.visual_review:
        config["dataset"].update(
            {"num_episodes": 12, "trajectories_per_layout": 1, "frames_per_trajectory": 1}
        )
        config["review"] = {"enabled": True, "index": 0}
    total_requested = (
        int(config["dataset"]["num_episodes"])
        * int(config["dataset"]["trajectories_per_layout"])
        * int(config["dataset"]["frames_per_trajectory"])
    )
    if total_requested > 12 and not args.visual_approved:
        raise SystemExit(
            "Visual-approval gate: runs larger than 12 frames require --visual-approved after reviewer approval"
        )
    output = Path(
        args.output or ("visual_review" if args.visual_review else config["dataset"]["output_dir"])
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    blender_app = Path(args.blender_app).resolve()
    blender_python = detect_blender_python_version(blender_app)
    shim = make_blenderproc_shim(blender_app)
    blenderproc = PROJECT_ROOT / ".venv" / "bin" / "blenderproc"
    if not blenderproc.is_file():
        raise FileNotFoundError(
            "Local BlenderProc missing. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        )
    runtime_dir = PROJECT_ROOT / ".runtime"
    runtime_dir.mkdir(exist_ok=True)
    runtime_path = runtime_dir / f"run_{uuid.uuid4().hex}.json"
    runtime = {
        "config": config,
        "output_root": str(output),
        "overwrite": bool(args.overwrite),
        "resume": bool(args.resume),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    env = dict(os.environ, IMAGINATION_RUNTIME_CONFIG=str(runtime_path))
    command = [
        str(blenderproc),
        "run",
        "--custom-blender-path",
        str(shim),
        str(PROJECT_ROOT / "generator" / "blender_entry.py"),
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    finally:
        runtime_path.unlink(missing_ok=True)
    if result.returncode != 0:
        return result.returncode
    elapsed = time.perf_counter() - started
    total_frames = (
        int(config["dataset"]["num_episodes"])
        * int(config["dataset"]["trajectories_per_layout"])
        * int(config["dataset"]["frames_per_trajectory"])
    )
    total_bytes = directory_size(output)
    seconds_per_frame = elapsed / max(total_frames, 1)
    bytes_per_frame = total_bytes / max(total_frames, 1)
    report = {
        "total_frames": total_frames,
        "wall_seconds": elapsed,
        "seconds_per_frame": seconds_per_frame,
        "total_bytes": total_bytes,
        "bytes_per_frame": bytes_per_frame,
        "projections": {
            str(frames): {
                "hours": seconds_per_frame * frames / 3600.0,
                "storage_gib": bytes_per_frame * frames / (1024**3),
            }
            for frames in (10_000, 20_000, 40_000)
        },
        "command": command,
        "config_path": str(config_path),
        "output_root": str(output),
        "blender_bundled_python": blender_python,
    }
    (output / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
