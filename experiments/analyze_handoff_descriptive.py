#!/usr/bin/env python3
"""Descriptive statistics for the clean QPU handoff experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def descriptive_summary(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    shots_col: str,
    tvd_col: str,
) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_cols, key)}
        shots = pd.to_numeric(group[shots_col], errors="coerce").dropna()
        tvd = pd.to_numeric(group[tvd_col], errors="coerce").dropna()
        if len(shots) != len(group) or len(tvd) != len(group):
            raise ValueError("missing or non-numeric shots/TVD values in a descriptive group")
        row.update({
            "runs": int(len(group)),
            "shots_min": float(shots.min()),
            "shots_max": float(shots.max()),
            "shots_median": float(shots.median()),
            "shots_mean": float(shots.mean()),
            "shots_std": float(shots.std(ddof=1)),
            "tvd_min": float(tvd.min()),
            "tvd_max": float(tvd.max()),
            "tvd_median": float(tvd.median()),
            "tvd_mean": float(tvd.mean()),
            "tvd_std": float(tvd.std(ddof=1)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


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
    analysis_dir.mkdir(parents=True, exist_ok=True)

    no_failure = pd.read_csv(raw_dir / "no_failure_runs.csv")
    handoffs = pd.read_csv(raw_dir / "handoff_runs.csv")

    no_failure_stats = descriptive_summary(
        no_failure,
        ["backend"],
        "shots",
        "final_tvd_to_aer",
    )
    handoff_pair_stats = descriptive_summary(
        handoffs,
        ["source_backend", "target_backend"],
        "total_shots",
        "final_aggregated_tvd_to_aer",
    )
    handoff_pair_fraction_stats = descriptive_summary(
        handoffs,
        ["source_backend", "target_backend", "failure_fraction"],
        "total_shots",
        "final_aggregated_tvd_to_aer",
    )

    no_failure_stats.to_csv(
        analysis_dir / "no_failure_qpu_descriptive_stats.csv",
        index=False,
    )
    handoff_pair_stats.to_csv(
        analysis_dir / "handoff_pair_descriptive_stats.csv",
        index=False,
    )
    handoff_pair_fraction_stats.to_csv(
        analysis_dir / "handoff_pair_failure_descriptive_stats.csv",
        index=False,
    )

    if len(no_failure_stats) != len(config["backends"]):
        raise RuntimeError("unexpected number of no-failure backend groups")
    expected_pairs = len(config["backends"]) * (len(config["backends"]) - 1)
    if len(handoff_pair_stats) != expected_pairs:
        raise RuntimeError("unexpected number of directed handoff-pair groups")
    if len(handoff_pair_fraction_stats) != expected_pairs * len(config["failure_fractions"]):
        raise RuntimeError("unexpected number of pair/failure groups")

    print(no_failure_stats.to_string(index=False))
    print(handoff_pair_stats.to_string(index=False))


if __name__ == "__main__":
    main()
