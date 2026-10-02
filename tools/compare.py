"""Paired episode-level comparison; it never asserts significance by itself."""

from __future__ import annotations
import argparse, json
from pathlib import Path

from common.runtime import save_json
from common.training.metrics import paired_bootstrap_delta


def compare(analytical_path, baseline_path, output, fnr_margin=None):
    analytical = json.loads(Path(analytical_path).read_text())
    baseline = json.loads(Path(baseline_path).read_text())
    interval = paired_bootstrap_delta(
        analytical["per_episode"], baseline["per_episode"], "hazard_fnr"
    )
    result = {
        "analytical": analytical["model"],
        "baseline": baseline["model"],
        "hazard_fnr_delta": interval,
        "configured_non_inferiority_margin": fnr_margin,
        "numeric_margin_met": None
        if fnr_margin is None or interval["ci95"][1] is None
        else interval["ci95"][1] <= fnr_margin,
        "statistical_significance_claimed": False,
    }
    save_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("analytical")
    p.add_argument("baseline")
    p.add_argument("--output", required=True)
    p.add_argument("--fnr-margin", type=float)
    a = p.parse_args()
    print(json.dumps(compare(a.analytical, a.baseline, a.output, a.fnr_margin), indent=2))


if __name__ == "__main__":
    main()
