#!/usr/bin/env python3
"""
Analyze results produced by experiments/run_failover_experiments.py.

Raw files under <output_dir>/raw are never modified. Derived CSVs are written to
<output_dir>/analysis and plots to <output_dir>/plots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def deterministic_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(p) for p in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def bootstrap_median_ci(values: Sequence[float], reps: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.empty(reps, dtype=float)
    for i in range(reps):
        medians[i] = np.median(rng.choice(arr, size=arr.size, replace=True))
    return float(np.quantile(medians, alpha / 2)), float(np.quantile(medians, 1 - alpha / 2))


def summary_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for key, group in df.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = dict(zip(group_cols, key))
        row["runs"] = int(len(group))
        for metric in metrics:
            if metric not in group:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            low, high = bootstrap_median_ci(
                vals.to_numpy(),
                bootstrap_reps,
                deterministic_seed(seed, *key, metric),
            )
            row[f"median_{metric}"] = float(vals.median())
            row[f"mean_{metric}"] = float(vals.mean())
            row[f"q25_{metric}"] = float(vals.quantile(0.25))
            row[f"q75_{metric}"] = float(vals.quantile(0.75))
            row[f"p95_{metric}"] = float(vals.quantile(0.95))
            row[f"median_{metric}_ci_low"] = low
            row[f"median_{metric}_ci_high"] = high
        if "target_violation" in group:
            row["target_violation_rate"] = float(pd.to_numeric(group["target_violation"], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def derive_oracle_replacement_runs(
    single: pd.DataFrame,
    references: pd.DataFrame,
    random_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    if single.empty or references.empty:
        return pd.DataFrame()
    error_lookup = {
        (str(row.circuit_key), str(row.backend)): float(row.reference_tvd_to_aer)
        for row in references.itertuples(index=False)
    }
    rows: List[pd.DataFrame] = []
    source_keys = single[["circuit_key", "source_backend"]].drop_duplicates()
    for sk in source_keys.itertuples(index=False):
        circuit_key = str(sk.circuit_key)
        source = str(sk.source_backend)
        candidates = sorted(
            {
                str(b)
                for b in single.loc[
                    (single["circuit_key"] == circuit_key) & (single["source_backend"] == source),
                    "target_backend",
                ].unique()
            }
        )
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda b: error_lookup[(circuit_key, b)])
        chosen = {
            "most_reliable_remaining": [(0, ranked[0])],
            "least_reliable_remaining": [(0, ranked[-1])],
        }
        random_choices: List[Tuple[int, str]] = []
        for rep in range(random_repetitions):
            rng = random.Random(deterministic_seed(seed, circuit_key, source, rep))
            random_choices.append((rep, rng.choice(candidates)))
        chosen["random_remaining"] = random_choices

        source_frame = single[
            (single["circuit_key"] == circuit_key) & (single["source_backend"] == source)
        ]
        for condition, selections in chosen.items():
            for rep, target in selections:
                selected = source_frame[source_frame["target_backend"] == target].copy()
                selected["replacement_condition"] = condition
                selected["replacement_repetition"] = rep
                selected["selected_target_tvd_to_aer"] = error_lookup[(circuit_key, target)]
                rows.append(selected)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def pair_summary(single: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    metrics = [
        "final_tvd_to_aer",
        "delta_tvd_vs_no_failure",
        "physical_shots_total",
        "post_failure_shots",
        "shot_overhead_vs_no_failure",
        "discarded_or_downweighted_shots",
    ]
    return summary_table(
        single,
        ["source_backend", "target_backend", "policy", "failure_fraction"],
        metrics,
        reps,
        seed,
    )


def pair_predictor_correlations(single: pd.DataFrame) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    predictors = ["pair_reference_tvd", "target_tvd_to_aer", "quality_change", "source_tvd_to_aer"]
    outcomes = ["final_tvd_to_aer", "delta_tvd_vs_no_failure", "post_failure_shots", "shot_overhead_vs_no_failure"]
    for policy, group in single.groupby("policy"):
        for failure_fraction, gf in group.groupby("failure_fraction"):
            for predictor in predictors:
                for outcome in outcomes:
                    pair = gf[[predictor, outcome]].dropna()
                    rho = float(pair[predictor].corr(pair[outcome], method="spearman")) if len(pair) >= 3 else float("nan")
                    rows.append({
                        "policy": policy,
                        "failure_fraction": float(failure_fraction),
                        "predictor": predictor,
                        "outcome": outcome,
                        "spearman_rho": rho,
                        "n": int(len(pair)),
                    })
    return pd.DataFrame(rows)


def decay_sensitivity(single: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()
    mask = single["policy"].astype(str).str.startswith("fixed_decay_") | (single["policy"] == "shift_aware_decay")
    frame = single[mask].copy()
    return summary_table(
        frame,
        ["policy", "failure_fraction"],
        ["final_tvd_to_aer", "delta_tvd_vs_no_failure", "physical_shots_total", "post_failure_shots"],
        reps,
        seed,
    )


def save_line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    if df.empty or x not in df or y not in df:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, group_df in df.groupby(group, sort=True):
        curve = group_df.groupby(x)[y].median().sort_index()
        ax.plot(curve.index, curve.values, marker="o", label=str(name))
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_oracle_bar(
    oracle: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    if oracle.empty:
        return
    grouped = oracle.groupby(["replacement_condition", "policy"])[metric].median().unstack(0)
    if grouped.empty:
        return
    ax = grouped.plot(kind="bar", figsize=(10, 5.5))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_pair_heatmaps(single: pd.DataFrame, policies: Sequence[str], path_dir: Path) -> None:
    if single.empty:
        return
    path_dir.mkdir(parents=True, exist_ok=True)
    for policy in policies:
        frame = single[single["policy"] == policy]
        if frame.empty:
            continue
        for metric in ["delta_tvd_vs_no_failure", "shot_overhead_vs_no_failure"]:
            matrix = frame.pivot_table(
                index="source_backend",
                columns="target_backend",
                values=metric,
                aggfunc="median",
            )
            if matrix.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 6))
            image = ax.imshow(matrix.to_numpy(), aspect="auto")
            ax.set_xticks(np.arange(len(matrix.columns)))
            ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(matrix.index)))
            ax.set_yticklabels(matrix.index)
            ax.set_xlabel("Target QPU")
            ax.set_ylabel("Source QPU")
            ax.set_title(f"{policy}: median {metric.replace('_', ' ')}")
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(path_dir / f"handoff_{policy}_{metric}.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze StableShots QPU-failover results")
    parser.add_argument("--config", default="experiments/failover_config.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_dir = Path(str(cfg["output_dir"]))
    raw_dir = output_dir / "raw"
    analysis_dir = output_dir / "analysis"
    plots_dir = output_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    analysis_cfg = cfg["analysis"]
    reps = int(analysis_cfg["bootstrap_repetitions"])
    seed = int(analysis_cfg["bootstrap_seed"])

    refs = read_optional(raw_dir / "qpu_references.csv")
    no_failure = read_optional(raw_dir / "no_failure_runs.csv")
    fixed = read_optional(raw_dir / "fixed_shot_runs.csv")
    single = read_optional(raw_dir / "single_failure_runs.csv")
    multi = read_optional(raw_dir / "multi_failure_runs.csv")
    stochastic = read_optional(raw_dir / "stochastic_failure_runs.csv")

    if not no_failure.empty:
        summary_table(
            no_failure,
            ["backend"],
            ["final_tvd_to_aer", "shots"],
            reps,
            seed,
        ).to_csv(analysis_dir / "no_failure_summary.csv", index=False)

    if not fixed.empty:
        summary_table(
            fixed,
            ["shots"],
            ["final_tvd_to_aer"],
            reps,
            seed,
        ).to_csv(analysis_dir / "fixed_shot_summary.csv", index=False)

    if not single.empty:
        summary_table(
            single,
            ["policy", "failure_fraction"],
            [
                "final_tvd_to_aer",
                "delta_tvd_vs_no_failure",
                "physical_shots_total",
                "post_failure_shots",
                "shot_overhead_vs_no_failure",
                "discarded_or_downweighted_shots",
            ],
            reps,
            seed,
        ).to_csv(analysis_dir / "single_failure_summary.csv", index=False)

        pair_summary(single, reps, seed).to_csv(analysis_dir / "handoff_pair_summary.csv", index=False)
        pair_predictor_correlations(single).to_csv(
            analysis_dir / "handoff_predictor_correlations.csv", index=False
        )
        decay_sensitivity(single, reps, seed).to_csv(analysis_dir / "decay_sensitivity.csv", index=False)

        replacement_cfg = cfg["replacement_conditions"]
        oracle = derive_oracle_replacement_runs(
            single,
            refs,
            int(replacement_cfg["random_repetitions"]),
            int(replacement_cfg["random_seed"]),
        )
        oracle.to_csv(analysis_dir / "oracle_replacement_runs.csv", index=False)
        summary_table(
            oracle,
            ["replacement_condition", "policy", "failure_fraction"],
            [
                "final_tvd_to_aer",
                "delta_tvd_vs_no_failure",
                "physical_shots_total",
                "post_failure_shots",
                "shot_overhead_vs_no_failure",
            ],
            reps,
            seed,
        ).to_csv(analysis_dir / "oracle_replacement_summary.csv", index=False)

        save_line_plot(
            single,
            "failure_fraction",
            "final_tvd_to_aer",
            "policy",
            "Median TVD to Aer",
            "Single failure: accuracy vs failure location",
            plots_dir / "failure_fraction_accuracy.png",
        )
        save_line_plot(
            single,
            "failure_fraction",
            "shot_overhead_vs_no_failure",
            "policy",
            "Median shot overhead",
            "Single failure: shot overhead vs failure location",
            plots_dir / "failure_fraction_shot_overhead.png",
        )
        save_oracle_bar(
            oracle,
            "final_tvd_to_aer",
            "Oracle replacement conditions: median final TVD",
            "Median TVD to Aer",
            plots_dir / "oracle_replacement_accuracy.png",
        )
        save_oracle_bar(
            oracle,
            "shot_overhead_vs_no_failure",
            "Oracle replacement conditions: median shot overhead",
            "Median shot overhead",
            plots_dir / "oracle_replacement_shot_overhead.png",
        )
        save_pair_heatmaps(
            single,
            [str(x) for x in analysis_cfg["primary_policies_for_pair_heatmaps"]],
            plots_dir / "handoff_heatmaps",
        )

        shift = single[single["policy"] == "shift_aware_decay"].copy()
        if not shift.empty:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.scatter(shift["shift_corrected_tvd"], shift["adaptive_lambda"], alpha=0.25)
            ax.set_xlabel("Bootstrap-corrected handoff TVD")
            ax.set_ylabel("Adaptive retention lambda")
            ax.set_title("Shift-aware decay response")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots_dir / "shift_aware_lambda_response.png", dpi=180)
            plt.close(fig)

    if not multi.empty:
        summary_table(
            multi,
            ["sequence_condition", "policy"],
            ["final_tvd_to_aer", "physical_shots_total", "discarded_or_downweighted_shots", "actual_failures"],
            reps,
            seed,
        ).to_csv(analysis_dir / "multi_failure_summary.csv", index=False)
        save_line_plot(
            multi,
            "actual_failures",
            "final_tvd_to_aer",
            "policy",
            "Median TVD to Aer",
            "Multiple failures: accuracy by realized failure count",
            plots_dir / "multi_failure_accuracy.png",
        )

    if not stochastic.empty:
        summary_table(
            stochastic,
            ["failure_probability_per_batch", "sequence_condition", "policy"],
            ["final_tvd_to_aer", "physical_shots_total", "actual_failures"],
            reps,
            seed,
        ).to_csv(analysis_dir / "stochastic_failure_summary.csv", index=False)
        save_line_plot(
            stochastic,
            "failure_probability_per_batch",
            "final_tvd_to_aer",
            "policy",
            "Median TVD to Aer",
            "Stochastic failures: accuracy vs per-batch failure probability",
            plots_dir / "stochastic_failure_accuracy.png",
        )

    print(f"analysis written under {analysis_dir}")
    print(f"plots written under {plots_dir}")


if __name__ == "__main__":
    main()
