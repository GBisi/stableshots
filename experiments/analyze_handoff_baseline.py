#!/usr/bin/env python3
"""Analysis for the clean directed QPU-handoff baseline experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def deterministic_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_median_ci(
    values: np.ndarray,
    repetitions: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    medians = np.median(values[indices], axis=1)
    return (
        float(np.quantile(medians, alpha / 2)),
        float(np.quantile(medians, 1 - alpha / 2)),
    )


def summarize(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, group in frame.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = {
            column: value for column, value in zip(group_cols, key)
        }
        row["runs"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"median_{metric}"] = (
                float(np.median(finite)) if len(finite) else float("nan")
            )
            row[f"q25_{metric}"] = (
                float(np.quantile(finite, 0.25)) if len(finite) else float("nan")
            )
            row[f"q75_{metric}"] = (
                float(np.quantile(finite, 0.75)) if len(finite) else float("nan")
            )
            low, high = bootstrap_median_ci(
                finite,
                repetitions,
                deterministic_seed(seed, *key, metric),
            )
            row[f"median_{metric}_ci_low"] = low
            row[f"median_{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def save_pair_heatmap(
    summary: pd.DataFrame,
    failure_fraction: float,
    value_col: str,
    backends: Sequence[str],
    title: str,
    output: Path,
) -> None:
    subset = summary[np.isclose(summary["failure_fraction"], failure_fraction)]
    matrix = np.full((len(backends), len(backends)), np.nan)
    for source_index, source in enumerate(backends):
        for target_index, target in enumerate(backends):
            if source == target:
                continue
            row = subset[
                (subset["source_backend"] == source)
                & (subset["target_backend"] == target)
            ]
            if not row.empty:
                matrix[source_index, target_index] = float(row.iloc[0][value_col])

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(backends)), labels=backends, rotation=35, ha="right")
    ax.set_yticks(range(len(backends)), labels=backends)
    ax.set_xlabel("Target backend")
    ax.set_ylabel("Source backend")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/handoff_baseline_config.json"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    output_dir = Path(str(config["output_dir"]))
    raw_dir = output_dir / "raw"
    analysis_dir = output_dir / "analysis"
    plots_dir = output_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    no_failure = pd.read_csv(raw_dir / "no_failure_runs.csv")
    handoffs = pd.read_csv(raw_dir / "handoff_runs.csv")

    repetitions = int(config["analysis"]["bootstrap_repetitions"])
    seed = int(config["analysis"]["bootstrap_seed"])

    handoffs["abs_delta_tvd_failure_to_final"] = handoffs[
        "delta_tvd_failure_to_final"
    ].abs()
    handoffs["abs_delta_tvd_vs_source_no_failure"] = handoffs[
        "delta_tvd_vs_source_no_failure"
    ].abs()

    no_failure_summary = summarize(
        no_failure,
        ["backend"],
        ["shots", "final_tvd_to_aer"],
        repetitions,
        seed,
    )
    no_failure_summary.to_csv(
        analysis_dir / "no_failure_backend_summary.csv",
        index=False,
    )

    metrics = [
        "failure_shots",
        "failure_point_tvd_to_aer",
        "post_failure_shots",
        "post_failure_only_tvd_to_aer",
        "total_shots",
        "source_evidence_share",
        "target_evidence_share",
        "final_aggregated_tvd_to_aer",
        "delta_tvd_failure_to_final",
        "abs_delta_tvd_failure_to_final",
        "delta_tvd_vs_source_no_failure",
        "abs_delta_tvd_vs_source_no_failure",
        "delta_tvd_vs_target_no_failure",
        "shot_delta_vs_source_no_failure",
    ]
    pair_summary = summarize(
        handoffs,
        ["source_backend", "target_backend", "failure_fraction"],
        metrics,
        repetitions,
        seed,
    )
    pair_summary.to_csv(
        analysis_dir / "handoff_pair_fraction_summary.csv",
        index=False,
    )
    handoffs.to_csv(
        analysis_dir / "handoff_runs_enriched.csv",
        index=False,
    )

    backends = [str(value) for value in config["backends"]]
    fractions = [float(value) for value in config["failure_fractions"]]
    heatmap_specs = [
        (
            "median_final_aggregated_tvd_to_aer",
            "Final aggregated TVD to Aer",
            "final_tvd",
        ),
        (
            "median_total_shots",
            "Total physical shots",
            "total_shots",
        ),
        (
            "median_failure_point_tvd_to_aer",
            "TVD to Aer at failure point",
            "failure_point_tvd",
        ),
        (
            "median_post_failure_only_tvd_to_aer",
            "Target-only post-failure TVD to Aer",
            "target_only_tvd",
        ),
        (
            "median_delta_tvd_failure_to_final",
            "Final TVD minus failure-point TVD",
            "delta_failure_to_final",
        ),
    ]
    for fraction in fractions:
        fraction_label = str(fraction).replace(".", "p")
        for value_col, label, stem in heatmap_specs:
            save_pair_heatmap(
                pair_summary,
                fraction,
                value_col,
                backends,
                f"{label} at failure fraction {fraction:g}",
                plots_dir / f"{stem}_f{fraction_label}.png",
            )

    manifest = {
        "no_failure_rows": int(len(no_failure)),
        "handoff_rows": int(len(handoffs)),
        "pair_fraction_summary_rows": int(len(pair_summary)),
        "primary_unit": "directed source-target pair within failure fraction",
        "note": (
            "No signed TVD statistic is pooled across opposite directed backend "
            "pairs in the primary analysis."
        ),
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
