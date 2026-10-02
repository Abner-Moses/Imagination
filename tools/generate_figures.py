"""Create deterministic, dependency-light diagnostics from a cached IMF frame."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
from common.registry import ANALYTICAL_CHANNELS
from data.adapter import load_manifest, resolve_path
from tools.imf_figure_diagnostics import (
    attention_diagnostics,
    bayesian_evidence_demo,
    draw_covariance_ellipses,
)


SIGNED = {
    "Gx",
    "Gy",
    "HarrisResponse",
    "OpticalFlowU",
    "OpticalFlowV",
    "DepthGradientX",
    "DepthGradientY",
}


def _font(size: int = 15):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _unit_image(values: np.ndarray, name: str) -> Image.Image:
    plane = np.asarray(values, dtype=np.float32)
    if name in SIGNED:
        plane = (np.clip(plane, -1.0, 1.0) + 1.0) * 0.5
    else:
        plane = np.clip(plane, 0.0, 1.0)
    image = Image.fromarray(np.uint8(np.round(plane * 255)), mode="L")
    return image.resize((192, 192), Image.Resampling.NEAREST).convert("RGB")


def _save_svg(image: Image.Image, path: Path, title: str) -> None:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'viewBox="0 0 %d %d"><title>%s</title><image width="100%%" height="100%%" '
        'href="data:image/png;base64,%s"/></svg>\n'
        % (image.width, image.height, image.width, image.height, title, encoded),
        encoding="utf-8",
    )


def _panel(title: str, body: Image.Image | None, note: str | None = None) -> Image.Image:
    panel = Image.new("RGB", (220, 230), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((10, 6), title, fill="black", font=_font(13))
    if body is not None:
        panel.paste(body.resize((192, 192)), (14, 30))
    elif note:
        draw.multiline_text((15, 70), note, fill=(70, 70, 70), font=_font(12), spacing=5)
    return panel


def _flow_visualization(u: np.ndarray, v: np.ndarray, valid: bool) -> Image.Image | None:
    if not valid:
        return None
    import cv2

    u = np.asarray(u, np.float32)
    v = np.asarray(v, np.float32)
    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8(angle / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(np.clip(magnitude * 255, 0, 255))
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb).resize((192, 192), Image.Resampling.NEAREST)


def _save_formats(image: Image.Image, directory: Path, stem: str, title: str) -> None:
    image.save(directory / f"{stem}.png")
    image.save(directory / f"{stem}.pdf", "PDF", resolution=300)
    _save_svg(image, directory / f"{stem}.svg", title)


def _make_grid(panels: list[Image.Image], columns: int = 4) -> Image.Image:
    rows = (len(panels) + columns - 1) // columns
    grid = Image.new("RGB", (columns * 220, rows * 230), "#d8dde4")
    for index, panel in enumerate(panels):
        grid.paste(panel, ((index % columns) * 220, (index // columns) * 230))
    return grid


def _select_record(manifest: Path, sample_id: str | None) -> dict:
    records = load_manifest(manifest)
    for record in records:
        if record["sample_id"] == sample_id:
            return record
    if sample_id is None:
        cached = [
            row for row in records if (manifest.parent / row["analytical_path"]).resolve().is_file()
        ]
        if cached:
            return next((row for row in cached if row["frame_index"] > 0), cached[0])
    raise FileNotFoundError(
        f"No analytical cache found for sample {sample_id or '(first available)'}"
    )


def _map_figure(path: Path, output: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities = []
    if isinstance(payload, dict):
        for group in payload.values():
            if isinstance(group, list):
                entities.extend(group)
    else:
        entities = payload
    canvas = Image.new("RGB", (1000, 720), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (30, 20),
        "Predicted local semantic map (map-frame x/y; metric units)",
        fill="black",
        font=_font(22),
    )
    points = [
        row for row in entities if row.get("valid", row.get("resolved", True)) and row.get("xyz_m")
    ]
    points += [row for row in entities if row.get("resolved") and row.get("map_xyz_m")]
    if not points:
        draw.text(
            (40, 90),
            "No resolved XYZ entities in the supplied map export.",
            fill="#555",
            font=_font(17),
        )
    else:
        xy = np.asarray([row.get("xyz_m", row.get("map_xyz_m"))[:2] for row in points], np.float32)
        low, high = xy.min(axis=0), xy.max(axis=0)
        span = np.maximum(high - low, 1e-3)
        for row, value in zip(points, xy):
            x = 70 + int((value[0] - low[0]) / span[0] * 850)
            y = 650 - int((value[1] - low[1]) / span[1] * 520)
            label = row.get("semantic_class", row.get("class_name", "unknown"))
            color = {
                "track": "#b64c4c",
                "cone": "#e58c27",
                "rock": "#666666",
                "landing": "#3a9d63",
            }.get(label, "#3478b8")
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
            draw.text((x + 8, y - 8), label, fill="#222", font=_font(12))
        covariance_count = draw_covariance_ellipses(
            draw,
            points,
            low,
            span,
            left=70,
            top=130,
            width=850,
            height=520,
        )
        if covariance_count:
            draw.text(
                (30, 54),
                "Thin blue contours: 95% horizontal position covariance",
                fill="#326fc0",
                font=_font(15),
            )
    _save_formats(canvas, output, "map_local_semantic", "Predicted local semantic map")


def generate(
    manifest: str | Path,
    output: str | Path,
    sample_id: str | None = None,
    map_json: str | Path | None = None,
    attention_npz: str | Path | None = None,
    bayesian_demo: bool = False,
) -> Path:
    manifest_path = Path(manifest).resolve()
    output_dir = Path(output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    record = _select_record(manifest_path, sample_id)
    rgb_path = resolve_path(manifest_path.parent, record["rgb_path"])
    cache_path = resolve_path(manifest_path.parent, record["analytical_path"])
    with Image.open(rgb_path) as source:
        rgb = source.convert("RGB").resize((192, 192), Image.Resampling.BILINEAR)
    with np.load(cache_path, allow_pickle=False) as cache:
        features = cache["features"]
        validity = cache["validity"] > 0
        names = tuple(str(value) for value in cache["channel_names"].tolist())
        flow_valid = (
            bool(cache["dense_flow_valid"].item())
            if "dense_flow_valid" in cache
            else bool(validity[20:22].any())
        )
    if features.shape != (28, 32, 32) or names != ANALYTICAL_CHANNELS:
        raise ValueError("Figure input does not match the versioned 28-channel IMF registry")

    def gray(index: int) -> Image.Image:
        return _unit_image(features[index], ANALYTICAL_CHANNELS[index])

    hog_tiles = [gray(index) for index in range(6, 15)]
    hog = Image.new("RGB", (576, 576), "white")
    for index, tile in enumerate(hog_tiles):
        hog.paste(tile, ((index % 3) * 192, (index // 3) * 192))
    valid_grid = Image.new("RGB", (576, 576), "white")
    for index in range(28):
        tile = (
            Image.fromarray(validity[index].astype(np.uint8) * 255)
            .resize((192, 192), Image.Resampling.NEAREST)
            .convert("RGB")
        )
        valid_grid.paste(tile, ((index % 3) * 192, (index // 3) * 192))
    chroma = (
        Image.fromarray(
            np.uint8(np.clip(np.maximum(features[18], features[19]), 0, 1) * 255), mode="L"
        )
        .resize((192, 192), Image.Resampling.NEAREST)
        .convert("RGB")
    )
    depth_gradient = (
        Image.fromarray(
            np.uint8(np.clip(np.hypot(features[23], features[24]) / np.sqrt(2), 0, 1) * 255),
            mode="L",
        )
        .resize((192, 192), Image.Resampling.NEAREST)
        .convert("RGB")
    )
    flow = _flow_visualization(features[20], features[21], flow_valid)
    stage_images = [
        ("01_rgb", "RGB frame", rgb),
        ("02_luminance", "Y luminance", gray(0)),
        ("03_cb", "Cb chroma", gray(1)),
        ("04_cr", "Cr chroma", gray(2)),
        ("05_gx", "Gx derivative", gray(3)),
        ("06_gy", "Gy derivative", gray(4)),
        ("07_gradient_magnitude", "Gradient magnitude", gray(5)),
        ("08_hog_visualization", "HOG bins 0–8", hog),
        ("09_harris", "Harris response", gray(15)),
        ("10_canny", "Canny edge", gray(16)),
        ("11_contours", "Contour map", gray(17)),
        ("12_chroma_gradients", "Chroma gradients", chroma),
        ("13_optical_flow", "Dense optical flow", flow),
        ("14_sparse_correspondences", "Sparse correspondences", None),
        ("15_sparse_3d_points", "Sparse 3-D points", None),
        ("16_depth", "Depth", gray(22)),
        ("17_depth_gradient", "Depth gradient", depth_gradient),
        ("18_slope", "Surface slope", gray(25)),
        ("19_roughness", "Surface roughness", gray(26)),
        ("20_geometry_confidence", "Geometry support", gray(27)),
        ("21_validity_maps", "Validity maps", valid_grid),
    ]
    panels = []
    for stem, title, image in stage_images:
        note = None
        if stem == "13_optical_flow" and not flow_valid:
            note = "Unavailable: no valid\nconsecutive-frame flow"
        elif image is None:
            note = "Not exported in this cache\nNo correspondence is invented"
        panel = _panel(title, image, note)
        panels.append(panel)
        if image is not None:
            image.save(output_dir / f"{stem}.png")
        else:
            panel.save(output_dir / f"{stem}.png")
    pipeline = _make_grid(panels, columns=4)
    _save_formats(pipeline, output_dir, "imf_pipeline", f"IMF pipeline {record['sample_id']}")
    _save_formats(valid_grid, output_dir, "imf_validity", "IMF feature validity maps")
    output_files = [f"{stem}.png" for stem, _, _ in stage_images]
    if map_json:
        _map_figure(Path(map_json), output_dir)
    if attention_npz:
        attention_image = attention_diagnostics(attention_npz)
        _save_formats(
            attention_image,
            output_dir,
            "imf_attention_diagnostics",
            "Sparse IMF attention, metric affinity and visibility",
        )
    if bayesian_demo:
        belief_image = bayesian_evidence_demo()
        _save_formats(
            belief_image,
            output_dir,
            "imf_bayesian_evidence_demo",
            "Illustrative synthetic semantic evidence fusion",
        )
    metadata = {
        "sample_id": record["sample_id"],
        "source_rgb": str(rgb_path),
        "analytical_cache": str(cache_path),
        "channel_order": names,
        "dense_flow_valid": flow_valid,
        "sparse_correspondences": "not exported in this NPZ cache; placeholder panel explicitly marked unavailable",
        "sparse_3d_points": "not exported in this NPZ cache; placeholder panel explicitly marked unavailable",
        "individual_channel_pngs": output_files,
    }
    (output_dir / "figure_source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/all.jsonl")
    parser.add_argument("--sample")
    parser.add_argument("--output", default="artifacts/paper_figures/pre_extraction")
    parser.add_argument("--map-json")
    parser.add_argument(
        "--attention-npz", help="Optional capture from tools/stream_map.py --attention-debug-output"
    )
    parser.add_argument(
        "--bayesian-demo",
        action="store_true",
        help="Add a synthetic, explicitly illustrative evidence-fusion figure",
    )
    args = parser.parse_args()
    print(
        generate(
            args.manifest,
            args.output,
            args.sample,
            args.map_json,
            args.attention_npz,
            args.bayesian_demo,
        )
    )


if __name__ == "__main__":
    main()
