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
import os
import subprocess
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def auto_publish_results(
    cfg: Mapping[str, object],
    paths: Sequence[Path],
    stage: str,
) -> None:
    """Commit and push only generated result paths after a successful stage."""
    publish = cfg.get("git_publish", {})
    if not isinstance(publish, Mapping) or not bool(publish.get("enabled", False)):
        return

    disabled = os.getenv("STABLESHOTS_DISABLE_AUTO_PUSH", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        print("automatic result push disabled by STABLESHOTS_DISABLE_AUTO_PUSH", flush=True)
        return

    expected_branch = str(publish.get("branch", "")).strip()
    remote = str(publish.get("remote", "origin")).strip() or "origin"
    if not expected_branch:
        raise ValueError("git_publish.branch must be set when automatic publishing is enabled")

    try:
        root_text = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("automatic result push requires execution inside a Git repository") from exc

    repo_root = Path(root_text).resolve()
    current_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_branch != expected_branch:
        raise RuntimeError(
            f"refusing to auto-push results from branch {current_branch!r}; "
            f"configured branch is {expected_branch!r}"
        )

    relative_paths: List[str] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative_paths.append(str(resolved.relative_to(repo_root)))
        except ValueError as exc:
            raise RuntimeError(f"result path {resolved} is outside repository {repo_root}") from exc

    subprocess.run(["git", "add", "--", *relative_paths], cwd=repo_root, check=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *relative_paths],
        cwd=repo_root,
        check=False,
    )
    if diff.returncode == 0:
        print(f"no new {stage} result changes to publish", flush=True)
        return
    if diff.returncode != 1:
        raise RuntimeError(f"git diff failed while preparing automatic {stage} result publication")

    message_key = f"{stage}_commit_message"
    default_message = f"Add failover {stage} results"
    message = str(publish.get(message_key, default_message)).strip() or default_message
    subprocess.run(
        ["git", "commit", "-m", message, "--", *relative_paths],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "push", remote, f"HEAD:refs/heads/{expected_branch}"],
        cwd=repo_root,
        check=True,
    )
    print(
        f"published {stage} results to {remote}/{expected_branch}",
        flush=True,
    )


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
    # Vectorize bootstrap draws in bounded chunks. This preserves the same
    # bootstrap estimator while avoiding millions of Python-level sampling loops
    # for the pairwise and oracle summaries.
    max_draws_per_chunk = 2_000_000
    chunk_reps = max(1, min(reps, max_draws_per_chunk // max(1, arr.size)))
    offset = 0
    while offset < reps:
        current = min(chunk_reps, reps - offset)
        samples = rng.choice(arr, size=(current, arr.size), replace=True)
        medians[offset:offset + current] = np.median(samples, axis=1)
        offset += current
    return float(np.quantile(medians, alpha / 2)), float(np.quantile(medians, 1 - alpha / 2))


def bootstrap_median_ci_clustered(
    group: pd.DataFrame,
    metric: str,
    cluster_col: str,
    reps: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    work = group[[cluster_col, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work[np.isfinite(work[metric].to_numpy(dtype=float))]
    work = work.dropna(subset=[cluster_col])
    if work.empty:
        return float("nan"), float("nan")

    clusters = list(work[cluster_col].drop_duplicates())
    if len(clusters) < 2:
        return bootstrap_median_ci(work[metric].to_numpy(), reps, seed, alpha)

    values_by_cluster = {
        cluster: work.loc[work[cluster_col] == cluster, metric].to_numpy(dtype=float)
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    medians = np.empty(reps, dtype=float)
    for rep in range(reps):
        sampled_indices = rng.integers(0, len(clusters), size=len(clusters))
        sampled_values = np.concatenate(
            [values_by_cluster[clusters[index]] for index in sampled_indices]
        )
        medians[rep] = np.median(sampled_values)
    return float(np.quantile(medians, alpha / 2)), float(np.quantile(medians, 1 - alpha / 2))


def summary_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
    bootstrap_reps: int,
    seed: int,
    cluster_col: str | None = None,
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
            ci_seed = deterministic_seed(seed, *key, metric)
            if cluster_col and cluster_col in group.columns:
                low, high = bootstrap_median_ci_clustered(
                    group,
                    metric,
                    cluster_col,
                    bootstrap_reps,
                    ci_seed,
                )
            else:
                low, high = bootstrap_median_ci(
                    vals.to_numpy(),
                    bootstrap_reps,
                    ci_seed,
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


def enrich_single_failure(
    single: pd.DataFrame,
    min_abs_restart_delta: float,
) -> pd.DataFrame:
    """Add direction, magnitude, and restart-normalized handoff-response metrics."""
    if single.empty:
        return single.copy()

    frame = single.copy()
    derived_columns = [
        "abs_delta_tvd",
        "handoff_direction",
        "restart_delta_tvd",
        "restart_abs_delta_tvd",
        "handoff_transfer_defined",
        "handoff_transfer_coeff",
        "handoff_transfer_abs",
        "handoff_transfer_outside_unit_interval",
        "target_evidence_share",
        "retained_source_evidence_share",
        "handoff_response_minus_target_evidence_share",
    ]
    frame = frame.drop(
        columns=[column for column in derived_columns if column in frame.columns],
        errors="ignore",
    )
    frame["abs_delta_tvd"] = frame["delta_tvd_vs_no_failure"].abs()
    effective_mass = pd.to_numeric(frame["effective_retained_shots"], errors="coerce")
    post_failure_mass = pd.to_numeric(frame["post_failure_shots"], errors="coerce")
    frame["target_evidence_share"] = np.where(
        effective_mass > 0,
        post_failure_mass / effective_mass,
        np.nan,
    )
    frame["retained_source_evidence_share"] = np.where(
        np.isfinite(frame["target_evidence_share"]),
        1.0 - frame["target_evidence_share"],
        np.nan,
    )
    frame["handoff_direction"] = np.where(
        frame["quality_change"] < 0,
        "improving",
        np.where(frame["quality_change"] > 0, "degrading", "near_equal"),
    )

    key_cols = ["circuit_key", "source_backend", "target_backend", "failure_fraction"]
    restart = (
        frame.loc[frame["policy"] == "full_restart", key_cols + ["delta_tvd_vs_no_failure"]]
        .rename(columns={"delta_tvd_vs_no_failure": "restart_delta_tvd"})
        .drop_duplicates(subset=key_cols)
    )
    frame = frame.merge(restart, on=key_cols, how="left", validate="many_to_one")
    frame["restart_abs_delta_tvd"] = frame["restart_delta_tvd"].abs()
    frame["handoff_transfer_defined"] = (
        frame["restart_abs_delta_tvd"] >= float(min_abs_restart_delta)
    )
    frame["handoff_transfer_coeff"] = np.where(
        frame["handoff_transfer_defined"],
        frame["delta_tvd_vs_no_failure"] / frame["restart_delta_tvd"],
        np.nan,
    )
    frame["handoff_transfer_abs"] = np.where(
        frame["handoff_transfer_defined"],
        frame["abs_delta_tvd"] / frame["restart_abs_delta_tvd"],
        np.nan,
    )
    frame["handoff_transfer_outside_unit_interval"] = np.where(
        frame["handoff_transfer_defined"],
        (frame["handoff_transfer_coeff"] < 0) | (frame["handoff_transfer_coeff"] > 1),
        False,
    )
    frame["handoff_response_minus_target_evidence_share"] = np.where(
        frame["handoff_transfer_defined"],
        frame["handoff_transfer_coeff"] - frame["target_evidence_share"],
        np.nan,
    )
    return frame


def handoff_transfer_summary(
    single: pd.DataFrame,
    group_cols: Sequence[str],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()

    summary = summary_table(
        single,
        group_cols,
        [
            "delta_tvd_vs_no_failure",
            "abs_delta_tvd",
            "handoff_transfer_coeff",
            "handoff_transfer_abs",
            "target_evidence_share",
            "handoff_response_minus_target_evidence_share",
            "physical_shots_total",
            "post_failure_shots",
            "shot_overhead_vs_no_failure",
        ],
        reps,
        seed,
        cluster_col="circuit_key",
    )

    extras: List[Dict[str, object]] = []
    for key, group in single.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        defined = group["handoff_transfer_defined"].astype(bool)
        defined_group = group.loc[defined]
        extras.append({
            **dict(zip(group_cols, key)),
            "transfer_defined_runs": int(defined.sum()),
            "transfer_defined_rate": float(defined.mean()),
            "transfer_outside_unit_interval_rate": (
                float(defined_group["handoff_transfer_outside_unit_interval"].mean())
                if len(defined_group)
                else float("nan")
            ),
        })
    return summary.merge(pd.DataFrame(extras), on=list(group_cols), how="left")


def transfer_threshold_sensitivity(
    single: pd.DataFrame,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        enriched = enrich_single_failure(single, float(threshold))
        for (policy, failure_fraction), group in enriched.groupby(
            ["policy", "failure_fraction"], sort=True
        ):
            defined = group[group["handoff_transfer_defined"]]
            rows.append({
                "min_abs_restart_delta": float(threshold),
                "policy": policy,
                "failure_fraction": float(failure_fraction),
                "runs": int(len(group)),
                "defined_runs": int(len(defined)),
                "defined_rate": float(len(defined) / len(group)) if len(group) else float("nan"),
                "median_handoff_transfer_coeff": (
                    float(defined["handoff_transfer_coeff"].median())
                    if len(defined)
                    else float("nan")
                ),
                "mean_handoff_transfer_coeff": (
                    float(defined["handoff_transfer_coeff"].mean())
                    if len(defined)
                    else float("nan")
                ),
                "median_abs_delta_tvd": float(group["abs_delta_tvd"].median()),
            })
    return pd.DataFrame(rows)


def paired_policy_comparison(
    single: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    reps: int,
    seed: int,
) -> Dict[str, object]:
    key_cols = ["circuit_key", "source_backend", "target_backend", "failure_fraction"]
    cols = key_cols + ["final_tvd_to_aer", "physical_shots_total"]
    left = single.loc[single["policy"] == policy_a, cols].rename(columns={
        "final_tvd_to_aer": "tvd_a",
        "physical_shots_total": "shots_a",
    })
    right = single.loc[single["policy"] == policy_b, cols].rename(columns={
        "final_tvd_to_aer": "tvd_b",
        "physical_shots_total": "shots_b",
    })
    pair = left.merge(right, on=key_cols, how="inner", validate="one_to_one")
    pair["tvd_diff"] = pair["tvd_a"] - pair["tvd_b"]
    pair["shot_diff"] = pair["shots_a"] - pair["shots_b"]
    tvd_diff = pair["tvd_diff"]
    shot_diff = pair["shot_diff"]
    tvd_ci_low, tvd_ci_high = bootstrap_median_ci_clustered(
        pair,
        "tvd_diff",
        "circuit_key",
        reps,
        deterministic_seed(seed, policy_a, policy_b, "paired_tvd"),
    )
    shot_ci_low, shot_ci_high = bootstrap_median_ci_clustered(
        pair,
        "shot_diff",
        "circuit_key",
        reps,
        deterministic_seed(seed, policy_a, policy_b, "paired_shots"),
    )
    tol = 1e-12
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "paired_runs": int(len(pair)),
        "median_tvd_difference_a_minus_b": float(tvd_diff.median()),
        "mean_tvd_difference_a_minus_b": float(tvd_diff.mean()),
        "median_tvd_difference_ci_low": tvd_ci_low,
        "median_tvd_difference_ci_high": tvd_ci_high,
        "median_shot_difference_a_minus_b": float(shot_diff.median()),
        "median_shot_difference_ci_low": shot_ci_low,
        "median_shot_difference_ci_high": shot_ci_high,
        "identical_tvd_and_shots": int(
            ((tvd_diff.abs() <= tol) & (shot_diff.abs() <= tol)).sum()
        ),
        "a_lower_tvd_runs": int((tvd_diff < -tol).sum()),
        "a_higher_tvd_runs": int((tvd_diff > tol).sum()),
    }


def pair_summary(single: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    metrics = [
        "final_tvd_to_aer",
        "delta_tvd_vs_no_failure",
        "abs_delta_tvd",
        "handoff_transfer_coeff",
        "target_evidence_share",
        "handoff_response_minus_target_evidence_share",
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
        cluster_col="circuit_key",
    )


def pair_predictor_correlations(single: pd.DataFrame) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    predictors = ["pair_reference_tvd", "target_tvd_to_aer", "quality_change", "source_tvd_to_aer"]
    outcomes = [
        "final_tvd_to_aer",
        "delta_tvd_vs_no_failure",
        "abs_delta_tvd",
        "handoff_transfer_coeff",
        "target_evidence_share",
        "handoff_response_minus_target_evidence_share",
        "post_failure_shots",
        "shot_overhead_vs_no_failure",
    ]
    for policy, group in single.groupby("policy"):
        for failure_fraction, gf in group.groupby("failure_fraction"):
            for predictor in predictors:
                for outcome in outcomes:
                    pair = gf[[predictor, outcome]].dropna()
                    if len(pair) >= 3:
                        # Spearman rho is Pearson correlation on ranks. Computing
                        # it explicitly keeps the analysis self-contained and
                        # avoids pandas' optional SciPy dependency.
                        x_rank = pair[predictor].rank(method="average")
                        y_rank = pair[outcome].rank(method="average")
                        rho = float(x_rank.corr(y_rank, method="pearson"))
                    else:
                        rho = float("nan")
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
        [
            "final_tvd_to_aer",
            "delta_tvd_vs_no_failure",
            "abs_delta_tvd",
            "handoff_transfer_coeff",
            "target_evidence_share",
            "handoff_response_minus_target_evidence_share",
            "physical_shots_total",
            "post_failure_shots",
        ],
        reps,
        seed,
        cluster_col="circuit_key",
    )


def evidence_share_correlations(single: pd.DataFrame) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    defined = single[single["handoff_transfer_defined"]].copy()
    for (policy, failure_fraction), group in defined.groupby(
        ["policy", "failure_fraction"], sort=True
    ):
        pair = group[["target_evidence_share", "handoff_transfer_coeff"]].dropna()
        if len(pair) >= 3:
            x_rank = pair["target_evidence_share"].rank(method="average")
            y_rank = pair["handoff_transfer_coeff"].rank(method="average")
            rho = float(x_rank.corr(y_rank, method="pearson"))
        else:
            rho = float("nan")
        rows.append({
            "policy": policy,
            "failure_fraction": float(failure_fraction),
            "n": int(len(pair)),
            "spearman_target_evidence_share_vs_h": rho,
            "median_target_evidence_share": (
                float(pair["target_evidence_share"].median()) if len(pair) else float("nan")
            ),
            "median_handoff_transfer_coeff": (
                float(pair["handoff_transfer_coeff"].median()) if len(pair) else float("nan")
            ),
            "median_h_minus_target_evidence_share": (
                float((pair["handoff_transfer_coeff"] - pair["target_evidence_share"]).median())
                if len(pair)
                else float("nan")
            ),
        })
    return pd.DataFrame(rows)


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
        for metric in [
            "delta_tvd_vs_no_failure",
            "abs_delta_tvd",
            "handoff_transfer_coeff",
            "shot_overhead_vs_no_failure",
        ]:
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
    transfer_min_abs_restart_delta = float(
        analysis_cfg.get("transfer_min_abs_restart_delta", 0.01)
    )
    transfer_thresholds = [
        float(x)
        for x in analysis_cfg.get(
            "transfer_threshold_sensitivity",
            [0.005, 0.01, 0.02],
        )
    ]

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
            cluster_col="circuit_key",
        ).to_csv(analysis_dir / "no_failure_summary.csv", index=False)

    if not fixed.empty:
        summary_table(
            fixed,
            ["shots"],
            ["final_tvd_to_aer"],
            reps,
            seed,
            cluster_col="circuit_key",
        ).to_csv(analysis_dir / "fixed_shot_summary.csv", index=False)

    if not single.empty:
        single = enrich_single_failure(single, transfer_min_abs_restart_delta)
        single.to_csv(analysis_dir / "single_failure_enriched.csv", index=False)

        summary_table(
            single,
            ["policy", "failure_fraction"],
            [
                "final_tvd_to_aer",
                "delta_tvd_vs_no_failure",
                "abs_delta_tvd",
                "handoff_transfer_coeff",
                "target_evidence_share",
                "handoff_response_minus_target_evidence_share",
                "physical_shots_total",
                "post_failure_shots",
                "shot_overhead_vs_no_failure",
                "discarded_or_downweighted_shots",
            ],
            reps,
            seed,
            cluster_col="circuit_key",
        ).to_csv(analysis_dir / "single_failure_summary.csv", index=False)

        handoff_transfer_summary(
            single,
            ["policy", "failure_fraction"],
            reps,
            seed,
        ).to_csv(analysis_dir / "handoff_transfer_summary.csv", index=False)

        handoff_transfer_summary(
            single,
            ["handoff_direction", "policy", "failure_fraction"],
            reps,
            seed,
        ).to_csv(analysis_dir / "handoff_direction_summary.csv", index=False)

        transfer_threshold_sensitivity(
            single,
            transfer_thresholds,
        ).to_csv(analysis_dir / "handoff_transfer_threshold_sensitivity.csv", index=False)

        paired = pd.DataFrame([
            paired_policy_comparison(single, "controller_reset", "naive_continuation", reps, seed),
            paired_policy_comparison(single, "epoch_local", "full_restart", reps, seed),
            paired_policy_comparison(single, "fixed_decay_0.25", "full_restart", reps, seed),
            paired_policy_comparison(single, "fixed_decay_0.5", "full_restart", reps, seed),
            paired_policy_comparison(single, "fixed_decay_0.75", "full_restart", reps, seed),
            paired_policy_comparison(single, "shift_aware_decay", "controller_reset", reps, seed),
            paired_policy_comparison(single, "shift_aware_decay", "fixed_decay_0.5", reps, seed),
        ])
        paired.to_csv(analysis_dir / "paired_policy_comparisons.csv", index=False)

        pair_summary(single, reps, seed).to_csv(analysis_dir / "handoff_pair_summary.csv", index=False)
        pair_predictor_correlations(single).to_csv(
            analysis_dir / "handoff_predictor_correlations.csv", index=False
        )
        decay_sensitivity(single, reps, seed).to_csv(analysis_dir / "decay_sensitivity.csv", index=False)
        evidence_share_correlations(single).to_csv(
            analysis_dir / "evidence_share_correlations.csv", index=False
        )

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
                "abs_delta_tvd",
                "handoff_transfer_coeff",
                "target_evidence_share",
                "handoff_response_minus_target_evidence_share",
                "physical_shots_total",
                "post_failure_shots",
                "shot_overhead_vs_no_failure",
            ],
            reps,
            seed,
            cluster_col="circuit_key",
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
        save_line_plot(
            single,
            "failure_fraction",
            "abs_delta_tvd",
            "policy",
            "Median |Delta TVD|",
            "Single failure: handoff magnitude vs failure location",
            plots_dir / "failure_fraction_abs_delta_tvd.png",
        )
        save_line_plot(
            single,
            "failure_fraction",
            "target_evidence_share",
            "policy",
            "Median target evidence share",
            "Evidence composition: replacement-QPU share vs failure location",
            plots_dir / "failure_fraction_target_evidence_share.png",
        )
        transfer_defined = single[single["handoff_transfer_defined"]].copy()
        save_line_plot(
            transfer_defined,
            "failure_fraction",
            "handoff_transfer_coeff",
            "policy",
            "Median restart-normalized handoff response H",
            "History inertia: restart-normalized response vs failure location",
            plots_dir / "failure_fraction_handoff_transfer.png",
        )
        for direction in ["improving", "degrading"]:
            direction_frame = single[single["handoff_direction"] == direction].copy()
            save_line_plot(
                direction_frame,
                "failure_fraction",
                "delta_tvd_vs_no_failure",
                "policy",
                "Median Delta TVD vs no failure",
                f"{direction.title()} handoffs: signed effect vs failure location",
                plots_dir / f"failure_fraction_delta_tvd_{direction}.png",
            )
            direction_transfer = direction_frame[
                direction_frame["handoff_transfer_defined"]
            ].copy()
            save_line_plot(
                direction_transfer,
                "failure_fraction",
                "handoff_transfer_coeff",
                "policy",
                "Median restart-normalized handoff response H",
                f"{direction.title()} handoffs: restart-normalized response",
                plots_dir / f"failure_fraction_handoff_transfer_{direction}.png",
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

    scenario_order = ["ascending_reliability", "descending_reliability", "random"]

    if not multi.empty:
        summary_table(
            multi,
            ["sequence_condition", "policy"],
            ["final_tvd_to_aer", "physical_shots_total", "discarded_or_downweighted_shots", "actual_failures"],
            reps,
            seed,
            cluster_col="circuit_key",
        ).to_csv(analysis_dir / "multi_failure_summary.csv", index=False)
        summary_table(
            multi,
            ["sequence_condition", "actual_failures", "policy"],
            ["final_tvd_to_aer", "physical_shots_total", "discarded_or_downweighted_shots"],
            reps,
            seed,
            cluster_col="circuit_key",
        ).to_csv(analysis_dir / "multi_failure_by_actual_failures.csv", index=False)
        (plots_dir / "multi_failure_accuracy.png").unlink(missing_ok=True)
        for scenario in scenario_order:
            scenario_frame = multi[multi["sequence_condition"] == scenario].copy()
            scenario_label = scenario.replace("_", " ").title()
            save_line_plot(
                scenario_frame,
                "actual_failures",
                "final_tvd_to_aer",
                "policy",
                "Median TVD to Aer",
                f"Multiple failures - {scenario_label}: accuracy by realized failure count",
                plots_dir / f"multi_failure_accuracy_{scenario}.png",
            )
            save_line_plot(
                scenario_frame,
                "actual_failures",
                "physical_shots_total",
                "policy",
                "Median physical shots",
                f"Multiple failures - {scenario_label}: physical shots by realized failure count",
                plots_dir / f"multi_failure_shots_{scenario}.png",
            )

    if not stochastic.empty:
        summary_table(
            stochastic,
            ["sequence_condition", "failure_probability_per_batch", "policy"],
            ["final_tvd_to_aer", "physical_shots_total", "actual_failures"],
            reps,
            seed,
            cluster_col="circuit_key",
        ).to_csv(analysis_dir / "stochastic_failure_summary.csv", index=False)
        summary_table(
            stochastic,
            ["sequence_condition", "failure_probability_per_batch", "actual_failures", "policy"],
            ["final_tvd_to_aer", "physical_shots_total"],
            reps,
            seed,
            cluster_col="circuit_key",
        ).to_csv(analysis_dir / "stochastic_failure_by_actual_failures.csv", index=False)
        (plots_dir / "stochastic_failure_accuracy.png").unlink(missing_ok=True)
        for scenario in scenario_order:
            scenario_frame = stochastic[stochastic["sequence_condition"] == scenario].copy()
            scenario_label = scenario.replace("_", " ").title()
            save_line_plot(
                scenario_frame,
                "failure_probability_per_batch",
                "final_tvd_to_aer",
                "policy",
                "Median TVD to Aer",
                f"Stochastic failures - {scenario_label}: accuracy vs failure probability",
                plots_dir / f"stochastic_failure_accuracy_{scenario}.png",
            )
            save_line_plot(
                scenario_frame,
                "failure_probability_per_batch",
                "physical_shots_total",
                "policy",
                "Median physical shots",
                f"Stochastic failures - {scenario_label}: physical shots vs failure probability",
                plots_dir / f"stochastic_failure_shots_{scenario}.png",
            )

    print(f"analysis written under {analysis_dir}")
    print(f"plots written under {plots_dir}")
    auto_publish_results(cfg, [analysis_dir, plots_dir], "analysis")


if __name__ == "__main__":
    main()
