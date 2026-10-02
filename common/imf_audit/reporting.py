"""Reproducible IMF-LBA JSON/CSV/Markdown, graph, and figure outputs."""

from __future__ import annotations

import base64
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from common.registry import ANALYTICAL_CHANNELS

from .contracts import Decision, OperatorResult, OperatorSpec
from .registry import LBA_SCHEMA, dependency_edges


COLORS = {
    Decision.ANALYTICAL: "#39805f",
    Decision.LEARNING_CANDIDATE: "#b64c4c",
    Decision.UNDECIDED: "#d69b2d",
    Decision.DEPENDENCY_BLOCKED: "#6f7380",
}


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _save_formats(image: Image.Image, directory: Path, stem: str, title: str) -> None:
    image.save(directory / f"{stem}.png")
    image.save(directory / f"{stem}.pdf", "PDF", resolution=300)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    (directory / f"{stem}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{image.width}" height="{image.height}" '
        f'viewBox="0 0 {image.width} {image.height}"><title>{title}</title>'
        f'<image width="100%" height="100%" href="data:image/png;base64,{encoded}"/></svg>\n',
        encoding="utf-8",
    )


def _bar_figure(results: dict[str, OperatorResult], output: Path) -> None:
    width, row_height = 1100, 42
    image = Image.new("RGB", (width, 90 + row_height * len(results)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "IMF-LBA analytical sufficiency ratio", fill="black", font=_font(22))
    draw.text(
        (24, 51),
        "R = max(error/ε, uncertainty/υ, failure/φ, cost/κ); N/A means unfrozen or unmeasured",
        fill="#444",
        font=_font(13),
    )
    for index, (name, result) in enumerate(results.items()):
        y = 86 + index * row_height
        draw.text((24, y), name, fill="#222", font=_font(13))
        draw.rectangle((280, y, 980, y + 20), fill="#edf0f4")
        if result.ratio is not None:
            length = min(700, int(350 * result.ratio))
            draw.rectangle((280, y, 280 + length, y + 20), fill=COLORS[result.decision])
            draw.text((990, y), f"{result.ratio:.3f}", fill="#222", font=_font(12))
        else:
            draw.text((290, y + 2), "N/A", fill="#666", font=_font(11))
        draw.line((630, y - 2, 630, y + 22), fill="#111", width=1)
    _save_formats(image, output, "analytical_sufficiency_ratio", "Analytical Sufficiency Ratio")


def _component_figure(results: dict[str, OperatorResult], output: Path) -> None:
    names = ("normalized_error", "normalized_uncertainty", "normalized_failure", "normalized_cost")
    image = Image.new("RGB", (1150, 100 + len(results) * 44), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "Normalized sufficiency components", fill="black", font=_font(22))
    for column, name in enumerate(names):
        draw.text(
            (420 + column * 165, 55), name.replace("normalized_", ""), fill="#222", font=_font(12)
        )
    for row, (operator, result) in enumerate(results.items()):
        y = 85 + row * 44
        draw.text((24, y), operator, fill="#222", font=_font(13))
        for column, name in enumerate(names):
            value = getattr(result, name)
            text = "N/A" if value is None else f"{value:.3f}"
            draw.text(
                (420 + column * 165, y),
                text,
                fill="#555" if value is None else "#222",
                font=_font(12),
            )
    _save_formats(image, output, "normalized_components", "Normalized sufficiency components")


def _status_figure(results: dict[str, OperatorResult], output: Path) -> None:
    counts = {decision: 0 for decision in Decision}
    for result in results.values():
        counts[result.decision] += 1
    image = Image.new("RGB", (900, 470), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "IMF-LBA decision status", fill="black", font=_font(22))
    maximum = max(counts.values()) or 1
    for index, (decision, count) in enumerate(counts.items()):
        x = 80 + index * 200
        height = int(280 * count / maximum)
        draw.rectangle((x, 380 - height, x + 120, 380), fill=COLORS[decision])
        draw.text((x + 48, 385 - height), str(count), fill="#222", font=_font(16))
        draw.multiline_text(
            (x, 400), decision.value.replace("_", "\n"), fill="#222", font=_font(11), align="center"
        )
    _save_formats(image, output, "decision_status", "IMF-LBA decision status")


def _stability_figure(stability: list[dict], output: Path) -> None:
    image = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (30, 20), "Learning-boundary stability by calibration window", fill="black", font=_font(22)
    )
    draw.line((80, 420, 840, 420), fill="#222", width=2)
    draw.line((80, 80, 80, 420), fill="#222", width=2)
    finite = [row for row in stability if row["stability"] is not None]
    if not finite:
        draw.text(
            (170, 230),
            "No pair of windows has shared resolved decisions yet.",
            fill="#666",
            font=_font(16),
        )
    else:
        for index, row in enumerate(finite):
            x = 120 + index * max(1, 680 // max(1, len(finite)))
            y = 420 - int(320 * row["stability"])
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#3478b8")
            draw.text(
                (x - 25, 430),
                f"{row['from_window']}→{row['to_window']}",
                fill="#222",
                font=_font(11),
            )
            draw.text((x - 15, y - 25), f"{row['stability']:.2f}", fill="#222", font=_font(11))
    _save_formats(image, output, "mask_stability", "Learning-boundary stability")


def _cost_error_figure(results: dict[str, OperatorResult], output: Path) -> None:
    image = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "Analytical cost versus error", fill="black", font=_font(22))
    measured = [
        (name, result)
        for name, result in results.items()
        if result.cost_value is not None and result.error_value is not None
    ]
    if not measured:
        draw.text(
            (180, 240),
            "Deployment operator cost is NOT_MEASURED in cached audits.",
            fill="#666",
            font=_font(16),
        )
    else:
        max_x = max(result.cost_value for _, result in measured) or 1
        max_y = max(result.error_value for _, result in measured) or 1
        for name, result in measured:
            x = 80 + int(730 * result.cost_value / max_x)
            y = 440 - int(350 * result.error_value / max_y)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=COLORS[result.decision])
            draw.text((x + 8, y - 8), name, fill="#222", font=_font(11))
    _save_formats(image, output, "cost_vs_error", "Analytical cost versus error")


def _dependency_outputs(registry: dict[str, OperatorSpec], output: Path) -> None:
    edges = dependency_edges(registry)
    (output / "dependency_graph.json").write_text(
        json.dumps(
            {
                "nodes": list(registry),
                "edges": [{"from": left, "to": right} for left, right in edges],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = ["digraph IMF_LBA {", "  rankdir=LR;"]
    lines += [f'  "{left}" -> "{right}";' for left, right in edges]
    lines.append("}")
    (output / "dependency_graph.dot").write_text("\n".join(lines) + "\n", encoding="utf-8")

    depths: dict[str, int] = {}
    for name, spec in registry.items():
        depths[name] = (
            0
            if not spec.input_dependencies
            else 1 + max(depths.get(dep, 0) for dep in spec.input_dependencies)
        )
    columns: dict[int, list[str]] = {}
    for name, depth in depths.items():
        columns.setdefault(depth, []).append(name)
    height = max(620, 100 + max(len(names) for names in columns.values()) * 70)
    width = max(1200, 300 + max(columns) * 280)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (25, 18), "IMF-LBA analytical responsibility dependency graph", fill="black", font=_font(22)
    )
    positions = {}
    for depth, names in columns.items():
        x = 35 + depth * 280
        for index, name in enumerate(names):
            y = 75 + index * 70
            positions[name] = (x, y)
            draw.rounded_rectangle(
                (x, y, x + 225, y + 38), radius=5, outline="#5d6878", fill="#eef2f6"
            )
            draw.text((x + 8, y + 10), name, fill="#222", font=_font(11))
    for left, right in edges:
        if left not in positions or right not in positions:
            continue
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        draw.line((x1 + 225, y1 + 19, x2, y2 + 19), fill="#718096", width=2)
    _save_formats(image, output / "figures", "dependency_graph", "IMF-LBA dependency graph")


def _feature_contract(
    output: Path, registry: dict[str, OperatorSpec], results: dict[str, OperatorResult]
) -> None:
    owners = {feature: name for name, spec in registry.items() for feature in spec.output_features}
    rows = []
    for feature in ANALYTICAL_CHANNELS:
        responsibility = owners.get(feature, "UNREGISTERED")
        result = results.get(responsibility)
        decision = result.decision.value if result else Decision.UNDECIDED.value
        rows.append(
            {
                "feature": feature,
                "responsibility": responsibility,
                "status": decision,
                "R": None if result is None else result.ratio,
                "suggested_value": "MSK"
                if result
                and result.decision in {Decision.LEARNING_CANDIDATE, Decision.DEPENDENCY_BLOCKED}
                else "PRESERVE",
                "suggested_validity": 0
                if result
                and result.decision in {Decision.LEARNING_CANDIDATE, Decision.DEPENDENCY_BLOCKED}
                else "UNCHANGED",
            }
        )
    with (output / "audited_feature_contract.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Audited IMF feature contract",
        "",
        "This is a design suggestion only; the 28-channel cache is unchanged.",
        "",
        "| Feature | Responsibility | Status | R | Suggested contract |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        ratio = "N/A" if row["R"] is None else f"{row['R']:.3f}"
        contract = "MSK; validity=0" if row["suggested_value"] == "MSK" else "preserve"
        lines.append(
            f"| {row['feature']} | {row['responsibility']} | {row['status']} | {ratio} | {contract} |"
        )
    (output / "audited_feature_contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    output: Path,
    metadata: dict[str, Any],
    registry: dict[str, OperatorSpec],
    histories: dict[int, dict[str, OperatorResult]],
    stability: list[dict],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    figures.mkdir()
    final_window = max(histories)
    final = histories[final_window]
    operators = {name: result.to_dict() for name, result in final.items()}
    mask = {
        name: int(result.decision == Decision.LEARNING_CANDIDATE)
        for name, result in final.items()
        if result.decision in {Decision.ANALYTICAL, Decision.LEARNING_CANDIDATE}
    }
    contract = {
        "schema": LBA_SCHEMA,
        "metadata": metadata,
        "window_size": final_window,
        "operators": operators,
        "learning_necessity_mask": mask,
        "decision_history": {
            name: [
                {
                    "window": window,
                    "decision": results[name].decision.value,
                    "R": results[name].ratio,
                }
                for window, results in histories.items()
            ]
            for name in registry
        },
    }
    (output / "learning_boundary.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "operator",
        "decision",
        "R",
        "ci_lower",
        "ci_upper",
        "evidence_type",
        "valid_samples",
        "coverage",
        "reasons",
        "blocked_by",
    ]
    with (output / "learning_boundary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, result in final.items():
            ci = result.confidence_interval or (None, None)
            writer.writerow(
                {
                    "operator": name,
                    "decision": result.decision.value,
                    "R": result.ratio,
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "evidence_type": result.evidence_type.value,
                    "valid_samples": result.valid_samples,
                    "coverage": result.coverage,
                    "reasons": ";".join(result.decision_reasons),
                    "blocked_by": ";".join(result.blocked_by),
                }
            )
    metric_fields = [
        "operator",
        "error",
        "error_limit",
        "normalized_error",
        "uncertainty",
        "uncertainty_limit",
        "normalized_uncertainty",
        "failure",
        "failure_limit",
        "normalized_failure",
        "cost",
        "cost_limit",
        "normalized_cost",
    ]
    with (output / "operator_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(metric_fields)
        for name, result in final.items():
            writer.writerow(
                [
                    name,
                    result.error_value,
                    result.error_threshold,
                    result.normalized_error,
                    result.uncertainty_value,
                    result.uncertainty_threshold,
                    result.normalized_uncertainty,
                    result.failure_rate,
                    result.failure_threshold,
                    result.normalized_failure,
                    result.cost_value,
                    result.cost_threshold,
                    result.normalized_cost,
                ]
            )
    with (output / "mask_stability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("from_window", "to_window", "stability", "resolved_fraction")
        )
        writer.writeheader()
        writer.writerows(stability)
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    specs = {
        name: {
            **asdict(spec),
            "reference_type": spec.reference_type.value,
            "thresholds": {key: asdict(value) for key, value in spec.thresholds.items()},
        }
        for name, spec in registry.items()
    }
    (output / "operator_registry.json").write_text(
        json.dumps(specs, indent=2) + "\n", encoding="utf-8"
    )
    suggestions = {}
    for name, result in final.items():
        if result.decision not in {Decision.LEARNING_CANDIDATE, Decision.DEPENDENCY_BLOCKED}:
            continue
        spec = registry[name]
        suggestions[name] = {
            "analytical_outputs": {
                feature: {"value": "MSK", "valid": False} for feature in spec.output_features
            },
            "available_upstream_inputs": list(spec.available_upstream_inputs),
            "blocked_downstream": list(spec.downstream_dependencies),
            "action": "HUMAN_REVIEW_REQUIRED",
        }
    (output / "suggested_msk_contract.json").write_text(
        json.dumps(
            {
                "schema": "imagination-msk-suggestion-v1",
                "automatic_model_mutation": False,
                "responsibilities": suggestions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _feature_contract(output, registry, final)
    _dependency_outputs(registry, output)
    lines = [
        "# IMF Learning-Boundary Audit",
        "",
        f"Calibration window: **{final_window} frames**",
        "",
        "| Responsibility | Decision | Evidence | R | Reason |",
        "|---|---|---|---:|---|",
    ]
    for name, result in final.items():
        ratio = "N/A" if result.ratio is None else f"{result.ratio:.3f}"
        lines.append(
            f"| {name} | {result.decision.value} | {result.evidence_type.value} | {ratio} | {', '.join(result.decision_reasons) or '—'} |"
        )
    lines += [
        "",
        "`MSK` means a storage placeholder accompanied by validity 0; it is never a measured numeric zero.",
        "",
        "This report proposes no neural module and performs no architecture mutation or training.",
    ]
    (output / "learning_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _bar_figure(final, figures)
    _component_figure(final, figures)
    _status_figure(final, figures)
    _stability_figure(stability, figures)
    _cost_error_figure(final, figures)
