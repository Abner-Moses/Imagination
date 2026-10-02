"""Focused diagnostics for map covariance, attention topology, and evidence."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_covariance_ellipses(
    draw: ImageDraw.ImageDraw,
    points,
    low,
    span,
    *,
    left=70,
    top=130,
    width=850,
    height=520,
    color="#326fc0",
) -> int:
    """Draw 95% 2-D covariance contours; return the count rendered."""
    rendered = 0
    for row in points:
        covariance = row.get("position_covariance_m2")
        if not row.get("covariance_valid") or covariance is None:
            continue
        matrix = np.asarray(covariance, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            continue
        matrix = (matrix[:2, :2] + matrix[:2, :2].T) * 0.5
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        except np.linalg.LinAlgError:
            continue
        if eigenvalues.min() < -1e-8:
            continue
        angles = np.linspace(0.0, 2.0 * math.pi, 49)
        circle = np.stack((np.cos(angles), np.sin(angles)))
        contour = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0) * 5.991)) @ circle
        map_xy = np.asarray(row.get("xyz_m", row.get("map_xyz_m"))[:2], dtype=np.float64)
        contour[0] += map_xy[0]
        contour[1] += map_xy[1]
        pixels = [
            (
                left + int((x - low[0]) / span[0] * width),
                top + height - int((y - low[1]) / span[1] * height),
            )
            for x, y in contour.T
        ]
        draw.line(pixels, fill=color, width=2)
        rendered += 1
    return rendered


def attention_diagnostics(npz_path: str | Path) -> Image.Image:
    """Render sparse neighborhoods, measured attention weights, metric affinity and visibility."""
    source = np.load(npz_path, allow_pickle=False)
    prefix = None
    for stage in ("stage3", "stage4"):
        if f"{stage}_block0_weights" in source.files:
            prefix = f"{stage}_block0"
            break
    if prefix is None:
        raise ValueError("Attention NPZ contains no stage3/stage4 block-0 weights")
    weights = np.asarray(source[f"{prefix}_weights"], dtype=np.float32).mean(axis=(0, 1))
    neighbor_key = f"{prefix}_neighbor_indices"
    if neighbor_key not in source.files:
        raise ValueError("This visualization expects gathered sparse attention neighbor indices")
    neighbors = np.asarray(source[neighbor_key])
    if neighbors.ndim == 2:
        neighbors = neighbors[None]
    neighbors = neighbors[0]
    grid_shape = tuple(int(value) for value in source[f"{prefix}_grid_shape"].tolist())
    height, width = grid_shape
    query = (height // 2) * width + width // 2
    k = neighbors.shape[-1]
    image = Image.new("RGB", (1500, 930), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (28, 16),
        f"Sparse IMF attention diagnostics — {prefix}, sample {source['sample_id'].item()}",
        fill="black",
        font=_font(22),
    )

    # Neighborhood graph in the stage grid.
    x0, y0, cell = 40, 90, 26
    selected = set(int(value) for value in neighbors[query])
    query_x, query_y = query % width, query // width
    for index in range(height * width):
        x, y = x0 + (index % width) * cell, y0 + (index // width) * cell
        center = (x + cell // 2, y + cell // 2)
        if index in selected and index != query:
            draw.line(
                (x0 + query_x * cell + cell // 2, y0 + query_y * cell + cell // 2, *center),
                fill="#9ebde3",
                width=1,
            )
        color = "#d14f45" if index == query else "#2468aa" if index in selected else "#d9dde2"
        draw.ellipse((x + 5, y + 5, x + cell - 5, y + cell - 5), fill=color)
    draw.text(
        (x0, y0 + height * cell + 12),
        f"Query cell {query}; selected K={k} grid/metric neighbors",
        fill="black",
        font=_font(14),
    )

    # Query's attention weights mapped back to image cells.
    heat_x, heat_y = 570, 90
    query_weights = weights[query]
    spatial_weights = query_weights[:k]
    weight_map = np.zeros((height, width), dtype=np.float32)
    for index, value in zip(neighbors[query], spatial_weights):
        weight_map[int(index) // width, int(index) % width] = value
    max_weight = max(float(weight_map.max()), 1e-9)
    tile = 26
    for y in range(height):
        for x in range(width):
            amount = float(weight_map[y, x] / max_weight)
            draw.rectangle(
                (
                    heat_x + x * tile,
                    heat_y + y * tile,
                    heat_x + (x + 1) * tile - 1,
                    heat_y + (y + 1) * tile - 1,
                ),
                fill=(int(245 - amount * 190), int(245 - amount * 95), 255),
            )
    draw.text(
        (heat_x, heat_y + height * tile + 12),
        "Query attention weight over selected spatial neighbors",
        fill="black",
        font=_font(14),
    )

    # Pairwise IMF Euclidean affinity, stored only on sparse edges.
    positions_key = f"{prefix}_positions"
    valid_key = f"{prefix}_position_validity"
    if positions_key in source.files and valid_key in source.files:
        positions = np.asarray(source[positions_key], dtype=np.float32)[0]
        validity = np.asarray(source[valid_key], dtype=np.float32)[0] > 0
        selected_positions = positions[neighbors]
        distances = np.linalg.norm(positions[:, None, :] - selected_positions, axis=-1)
        pair_valid = validity[:, None] & validity[neighbors]
        scale_key = f"{prefix}_metric_scale_m"
        scale = float(source[scale_key]) if scale_key in source.files else 4.0
        affinity = np.exp(-0.5 * np.square(distances / max(scale, 1e-6))) * pair_valid
        ax, ay, aw, ah = 40, 610, 780, 250
        canvas = Image.new("L", (affinity.shape[1], affinity.shape[0]), 0)
        canvas.putdata(np.uint8(np.clip(affinity.reshape(-1) * 255, 0, 255)))
        image.paste(canvas.resize((aw, ah), Image.Resampling.NEAREST).convert("RGB"), (ax, ay))
        draw.rectangle((ax, ay, ax + aw, ay + ah), outline="#333")
        draw.text(
            (ax, ay - 24),
            "IMF Euclidean affinity on selected sparse edges (invalid geometry = 0)",
            fill="black",
            font=_font(14),
        )

    context_key = f"{prefix}_context_validity"
    if context_key in source.files:
        context_validity = np.asarray(source[context_key]).reshape(-1)
        cx, cy, cw = 900, 620, 15
        for index, value in enumerate(context_validity[:32]):
            x, y = cx + (index % 16) * cw, cy + (index // 16) * cw
            draw.rectangle(
                (x, y, x + cw - 2, y + cw - 2), fill="#2a9d62" if value > 0 else "#d9534f"
            )
        draw.text((cx, cy + 42), "Prior-map visibility/support mask", fill="black", font=_font(14))
    return image


def bayesian_evidence_demo() -> Image.Image:
    """Show code-generated evidence fusion on a deterministic synthetic sequence."""
    from common.mapping.semantic_map import PersistentSemanticMap, SemanticObservation

    probability = (0.82, 0.72, 0.55, 0.68, 0.91, 0.76, 0.88, 0.64)
    traces = {}
    for fusion in ("arithmetic", "bayesian"):
        local_map = PersistentSemanticMap(semantic_fusion=fusion, evidence_correlation_time_s=1.0)
        trace = []
        for frame, track_probability in enumerate(probability):
            entity = local_map.update(
                SemanticObservation(
                    (0.0, 0.0, 0.0),
                    {"track": track_probability, "grass": 1.0 - track_probability},
                    0.9,
                    frame * 0.1,
                )
            )
            trace.append(entity.class_probabilities.get("track", 0.0))
        traces[fusion] = trace
    image = Image.new("RGB", (1100, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (28, 16),
        "Illustrative synthetic semantic evidence fusion (not trained-model data)",
        fill="black",
        font=_font(21),
    )
    left, top, width, height = 90, 80, 930, 390
    draw.rectangle((left, top, left + width, top + height), outline="#333")
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + height - int(fraction * height)
        draw.line((left, y, left + width, y), fill="#e4e6e8", width=1)
        draw.text((35, y - 8), f"{fraction:.2f}", fill="#333", font=_font(13))
    colors = {"arithmetic": "#dd8452", "bayesian": "#2868a6"}
    for name, values in traces.items():
        points = [
            (left + int(index * width / (len(values) - 1)), top + height - int(value * height))
            for index, value in enumerate(values)
        ]
        draw.line(points, fill=colors[name], width=4)
        for point in points:
            draw.ellipse(
                (point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=colors[name]
            )
    draw.text(
        (left, 490),
        "Observation index (fixed sequence of predicted class probabilities)",
        fill="black",
        font=_font(15),
    )
    draw.line((760, 520, 800, 520), fill=colors["arithmetic"], width=4)
    draw.text((808, 511), "Arithmetic reference", fill="black", font=_font(14))
    draw.line((930, 520, 970, 520), fill=colors["bayesian"], width=4)
    draw.text((978, 511), "Bounded evidence", fill="black", font=_font(14))
    return image
