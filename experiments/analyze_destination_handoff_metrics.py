#!/usr/bin/env python3
"""Destination-centric StableShots+keep comparison.

For every StableShots+keep handoff candidate (excluding the random circuit
family), compare the destination QPU with its matched no-failure StableShots
execution on the same circuit/size.

Two accounting conventions are produced for both resources and TVD:
1. source counted:
   - shots = source pre-failure + destination post-failure;
   - TVD = final aggregate estimator, containing source + destination counts.
2. source not counted:
   - shots = destination post-failure shots only;
   - TVD = TVD of destination post-failure counts only.

All deltas are handoff minus matched destination no-failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_handoff_baseline_experiment import StableShotsConfig, materialize_circuit, tvd_weighted
from run_handoff_policy_matrix import take_exact_shots


def load_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def paired_boxplot(
    ax: plt.Axes,
    frame: pd.DataFrame,
    fractions: List[float],
    left: str,
    right: str,
    ylabel: str,
    left_label: str,
    right_label: str,
) -> None:
    arrays = []
    positions = []
    centers = []
    for i, fraction in enumerate(fractions):
        center = i * 3.0
        centers.append(center)
        g = frame[np.isclose(frame["failure_fraction"], fraction)]
        arrays.extend([
            g[left].dropna().to_numpy(dtype=float),
            g[right].dropna().to_numpy(dtype=float),
        ])
        positions.extend([center - 0.45, center + 0.45])
    ax.boxplot(arrays, positions=positions, widths=0.7, showfliers=False)
    ax.axhline(0.0, linewidth=0.9)
    ax.set_xticks(centers, [f"{f:g}" for f in fractions])
    ax.set_xlabel("Failure fraction")
    ax.set_ylabel(ylabel)
    ax.plot([], [], label=left_label)
    ax.plot([], [], label=right_label)
    ax.legend()


def destination_panels(
    frame: pd.DataFrame,
    backends: List[str],
    fractions: List[float],
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2))
    axes_flat = list(axes.flat)
    for ax, target in zip(axes_flat, backends):
        work = frame[frame["target_backend"] == target]
        arrays = [
            work.loc[
                np.isclose(work["failure_fraction"], fraction),
                metric,
            ].dropna().to_numpy(dtype=float)
            for fraction in fractions
        ]
        ax.boxplot(
            arrays,
            tick_labels=[f"{f:g}" for f in fractions],
            showfliers=False,
        )
        ax.axhline(0.0, linewidth=0.9)
        ax.set_title(target.replace("fake_", ""))
        ax.set_xlabel("Failure fraction")
        ax.set_ylabel(ylabel)
    axes_flat[-1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/handoff_policy_matrix_config.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qpu_handoff_policy_matrix/analysis_v5_destination"),
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/qpu_handoff_policy_matrix/plots_v5_destination"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path(str(config["output_dir"])) / "raw"
    handoff = pd.read_csv(raw_dir / "handoff_policy_runs.csv")
    no_failure = pd.read_csv(raw_dir / "no_failure_policy_runs.csv")

    handoff = handoff[
        (handoff["policy_id"] == "stableshots_keep")
        & (handoff["algorithm"] != "random")
    ].copy()
    no_failure = no_failure[
        (no_failure["stopping_rule"] == "stableshots")
        & (no_failure["algorithm"] != "random")
    ].copy()

    stable_raw = config["stableshots"]
    stable_config = StableShotsConfig(
        batch_size=int(stable_raw["batch_size"]),
        lookback_batches=int(stable_raw["lookback_batches"]),
        stability=int(stable_raw["stability"]),
        epsilon=float(stable_raw["epsilon"]),
        max_shots=int(stable_raw["max_shots"]),
    )

    backends = [str(x) for x in config["backends"]]
    fractions = [float(x) for x in config["failure_fractions"]]

    dest_nf = no_failure[
        [
            "circuit_key",
            "backend",
            "final_tvd_to_aer",
            "total_physical_shots",
        ]
    ].rename(
        columns={
            "backend": "target_backend",
            "final_tvd_to_aer": "destination_no_failure_tvd_to_aer",
            "total_physical_shots": "destination_no_failure_shots",
        }
    )

    work = handoff.merge(
        dest_nf,
        on=["circuit_key", "target_backend"],
        how="left",
        validate="many_to_one",
    )
    if work[
        ["destination_no_failure_tvd_to_aer", "destination_no_failure_shots"]
    ].isna().any().any():
        raise RuntimeError("failed to match destination no-failure baseline")

    destination_only_tvd: Dict[tuple, float] = {}

    for (algorithm, size), group in work.groupby(["algorithm", "size"], sort=True):
        print(f"materializing {algorithm}/{size}", flush=True)
        ideal, streams = materialize_circuit(
            str(algorithm),
            int(size),
            config,
            stable_config,
        )
        for row in group.itertuples():
            post = int(row.post_failure_shots)
            key = (
                str(row.circuit_key),
                str(row.source_backend),
                str(row.target_backend),
                float(row.failure_fraction),
            )
            if post <= 0:
                destination_only_tvd[key] = float("nan")
            else:
                counts = take_exact_shots(streams[str(row.target_backend)], post)
                destination_only_tvd[key] = float(tvd_weighted(counts, ideal))

    work["destination_only_post_tvd_to_aer"] = [
        destination_only_tvd[
            (
                str(r.circuit_key),
                str(r.source_backend),
                str(r.target_backend),
                float(r.failure_fraction),
            )
        ]
        for r in work.itertuples()
    ]

    # Shot deltas.
    work["delta_shots_source_counted"] = (
        work["total_physical_shots"].astype(float)
        - work["destination_no_failure_shots"].astype(float)
    )
    work["delta_shots_destination_only"] = (
        work["post_failure_shots"].astype(float)
        - work["destination_no_failure_shots"].astype(float)
    )

    # TVD deltas.
    work["delta_tvd_source_counted"] = (
        work["final_tvd_to_aer"].astype(float)
        - work["destination_no_failure_tvd_to_aer"].astype(float)
    )
    work["delta_tvd_destination_only"] = (
        work["destination_only_post_tvd_to_aer"].astype(float)
        - work["destination_no_failure_tvd_to_aer"].astype(float)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "circuit_key",
        "algorithm",
        "size",
        "source_backend",
        "target_backend",
        "failure_fraction",
        "failure_shots",
        "post_failure_shots",
        "total_physical_shots",
        "destination_no_failure_shots",
        "final_tvd_to_aer",
        "destination_only_post_tvd_to_aer",
        "destination_no_failure_tvd_to_aer",
        "delta_shots_source_counted",
        "delta_shots_destination_only",
        "delta_tvd_source_counted",
        "delta_tvd_destination_only",
    ]
    work[cols].to_csv(
        args.output_dir / "destination_handoff_comparison.csv",
        index=False,
    )

    # Destination-specific plots: shots.
    destination_panels(
        work,
        backends,
        fractions,
        "delta_shots_source_counted",
        "Delta shots vs destination no-failure",
        "StableShots + keep: handoff cost vs destination no-failure (source shots counted)",
        args.plots_dir / "destination_shots_source_counted.png",
    )
    destination_panels(
        work,
        backends,
        fractions,
        "delta_shots_destination_only",
        "Delta destination shots vs destination no-failure",
        "StableShots + keep: handoff cost vs destination no-failure (source shots not counted)",
        args.plots_dir / "destination_shots_destination_only.png",
    )

    # Destination-specific plots: TVD.
    destination_panels(
        work,
        backends,
        fractions,
        "delta_tvd_source_counted",
        "Delta TVD vs destination no-failure",
        "StableShots + keep: handoff TVD vs destination no-failure (source data counted)",
        args.plots_dir / "destination_tvd_source_counted.png",
    )
    destination_panels(
        work,
        backends,
        fractions,
        "delta_tvd_destination_only",
        "Delta destination-only TVD vs destination no-failure",
        "StableShots + keep: destination-only TVD vs no-failure (source data not counted)",
        args.plots_dir / "destination_tvd_destination_only.png",
    )

    # Aggregated paired plot: shots.
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    paired_boxplot(
        ax,
        work,
        fractions,
        "delta_shots_source_counted",
        "delta_shots_destination_only",
        "Delta shots vs matched destination no-failure",
        "Source + destination",
        "Destination only",
    )
    ax.set_title("StableShots + keep: aggregated destination shot impact")
    fig.tight_layout()
    fig.savefig(args.plots_dir / "aggregated_shot_delta.png", dpi=180)
    plt.close(fig)

    # Aggregated paired plot: TVD.
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    paired_boxplot(
        ax,
        work,
        fractions,
        "delta_tvd_source_counted",
        "delta_tvd_destination_only",
        "Delta TVD vs matched destination no-failure",
        "Source + destination estimator",
        "Destination-only estimator",
    )
    ax.set_title("StableShots + keep: aggregated destination TVD impact")
    fig.tight_layout()
    fig.savefig(args.plots_dir / "aggregated_tvd_delta.png", dpi=180)
    plt.close(fig)

    # Compact summaries.
    rows: List[Dict[str, object]] = []
    for fraction, group in work.groupby("failure_fraction", sort=True):
        row: Dict[str, object] = {
            "failure_fraction": float(fraction),
            "rows": int(len(group)),
        }
        for metric in [
            "delta_shots_source_counted",
            "delta_shots_destination_only",
            "delta_tvd_source_counted",
            "delta_tvd_destination_only",
        ]:
            s = group[metric].dropna().astype(float)
            row[f"{metric}_min"] = float(s.min())
            row[f"{metric}_q25"] = float(s.quantile(0.25))
            row[f"{metric}_median"] = float(s.median())
            row[f"{metric}_mean"] = float(s.mean())
            row[f"{metric}_q75"] = float(s.quantile(0.75))
            row[f"{metric}_max"] = float(s.max())
            row[f"{metric}_std"] = float(s.std(ddof=1))
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "destination_handoff_summary_by_failure.csv",
        index=False,
    )

    manifest = {
        "rows": int(len(work)),
        "algorithms": sorted(work["algorithm"].unique().tolist()),
        "backends": sorted(work["target_backend"].unique().tolist()),
        "failure_fractions": fractions,
        "destination_only_tvd_nan_rows": int(
            work["destination_only_post_tvd_to_aer"].isna().sum()
        ),
        "definitions": {
            "delta_shots_source_counted": (
                "total physical handoff shots - destination no-failure StableShots shots"
            ),
            "delta_shots_destination_only": (
                "destination post-handoff shots - destination no-failure StableShots shots"
            ),
            "delta_tvd_source_counted": (
                "final aggregate handoff TVD - destination no-failure StableShots TVD"
            ),
            "delta_tvd_destination_only": (
                "destination-only post-handoff TVD - destination no-failure StableShots TVD"
            ),
        },
    }
    (args.output_dir / "destination_handoff_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
