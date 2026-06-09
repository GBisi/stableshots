#!/usr/bin/env python3
"""Size-aware grid selection for StableShots multi-QPU grid-search outputs.

This script reuses an existing `grid_candidate_circuit_level_metrics.csv` file and
reruns only the train/test selection step. It does not materialize traces and does
not recompute policy/subset metrics.

Main purpose:
  --selection-risk-group size
  --selection-risk-aggregation max_group_q95

The selection risk is computed as:
  max_{size} q95(circuit_median_tvd within that size on the train split)

Then, for each (policy, scheme), the selector chooses the configuration satisfying
risk <= target with the best objective, by default maximum train median SSR. If no
candidate satisfies the constraint, selection is relaxed to the lowest risk.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_POLICIES = [
    "local_only_offline_aggregate",
    "aggregate_only_sync",
    "hierarchical_local_aggregate",
]
DEFAULT_SCHEMES = ["uniform", "shot_proportional", "confidence"]
DEFAULT_SIZES = list(range(4, 16))


def parse_csv_strings(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(x) for x in parse_csv_strings(value)]


def q95(values: pd.Series) -> float:
    return float(values.quantile(0.95)) if len(values) else float("nan")


def q75(values: pd.Series) -> float:
    return float(values.quantile(0.75)) if len(values) else float("nan")


def valid_col(target: float) -> str:
    s = f"{target:g}".replace(".", "p")
    return f"valid_rate_tvd_le_{s}"


def summarize_metric_frame(frame: pd.DataFrame, prefix: str, target_tvd: float) -> Dict[str, object]:
    if frame.empty:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_circuit_configs": 0,
            f"{prefix}_k_values": 0,
            f"{prefix}_median_tvd": np.nan,
            f"{prefix}_q75_tvd": np.nan,
            f"{prefix}_q95_tvd": np.nan,
            f"{prefix}_max_tvd": np.nan,
            f"{prefix}_{valid_col(0.01)}": np.nan,
            f"{prefix}_{valid_col(target_tvd)}": np.nan,
            f"{prefix}_{valid_col(0.10)}": np.nan,
            f"{prefix}_median_ssr": np.nan,
            f"{prefix}_q25_ssr": np.nan,
            f"{prefix}_median_parallel_ssr": np.nan,
            f"{prefix}_median_total_shots": np.nan,
            f"{prefix}_median_max_backend_shots": np.nan,
        }
    return {
        f"{prefix}_rows": int(len(frame)),
        f"{prefix}_circuit_configs": int(frame["circuit_key"].nunique()),
        f"{prefix}_k_values": int(frame["k"].nunique()),
        f"{prefix}_median_tvd": float(frame["circuit_median_tvd"].median()),
        f"{prefix}_q75_tvd": float(frame["circuit_median_tvd"].quantile(0.75)),
        f"{prefix}_q95_tvd": float(frame["circuit_median_tvd"].quantile(0.95)),
        f"{prefix}_max_tvd": float(frame["circuit_median_tvd"].max()),
        f"{prefix}_{valid_col(0.01)}": float((frame["circuit_median_tvd"] <= 0.01).mean()),
        f"{prefix}_{valid_col(target_tvd)}": float((frame["circuit_median_tvd"] <= target_tvd).mean()),
        f"{prefix}_{valid_col(0.10)}": float((frame["circuit_median_tvd"] <= 0.10).mean()),
        f"{prefix}_median_ssr": float(frame["circuit_median_ssr"].median()),
        f"{prefix}_q25_ssr": float(frame["circuit_median_ssr"].quantile(0.25)),
        f"{prefix}_median_parallel_ssr": float(frame["circuit_median_parallel_ssr"].median()),
        f"{prefix}_median_total_shots": float(frame["circuit_median_total_shots"].median()),
        f"{prefix}_median_max_backend_shots": float(frame["circuit_median_max_backend_shots"].median()),
    }


def compute_group_risk(
    frame: pd.DataFrame,
    group_col: str,
    aggregation: str,
    tvd_col: str = "circuit_median_tvd",
) -> Tuple[float, Dict[str, float]]:
    if frame.empty:
        return float("nan"), {}
    if group_col not in frame.columns:
        raise ValueError(f"risk group column {group_col!r} is not in candidate table")
    group_q95 = frame.groupby(group_col)[tvd_col].quantile(0.95).astype(float).to_dict()
    group_q95 = {str(k): float(v) for k, v in group_q95.items()}
    if aggregation == "max_group_q95":
        return float(max(group_q95.values())) if group_q95 else float("nan"), group_q95
    if aggregation == "mean_group_q95":
        return float(np.mean(list(group_q95.values()))) if group_q95 else float("nan"), group_q95
    if aggregation == "median_group_q95":
        return float(np.median(list(group_q95.values()))) if group_q95 else float("nan"), group_q95
    raise ValueError(f"unsupported selection-risk-aggregation {aggregation!r}")


def make_stratified_size_splits(
    circuits: pd.DataFrame,
    test_fraction: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    circuits = circuits[["circuit_key", "algorithm", "size", "circuit_kind"]].drop_duplicates().copy()
    for rep in range(repetitions):
        rng = np.random.default_rng(seed + rep)
        split_id = f"stratified_size_rep{rep:03d}"
        for size, group in circuits.groupby("size", sort=True):
            group = group.sort_values(["algorithm", "circuit_key"]).reset_index(drop=True)
            n = len(group)
            n_test = max(1, min(n - 1, int(round(n * test_fraction))))
            test_idx = set(int(i) for i in rng.choice(n, size=n_test, replace=False))
            for idx, row in group.iterrows():
                rows.append(
                    {
                        "split_id": split_id,
                        "split_mode": "stratified_size",
                        "repetition": rep,
                        "circuit_key": row["circuit_key"],
                        "split_role": "test" if idx in test_idx else "train",
                        "algorithm": row["algorithm"],
                        "size": int(row["size"]),
                        "circuit_kind": row["circuit_kind"],
                    }
                )
    return pd.DataFrame(rows)


def load_or_make_splits(input_dir: Path, candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    split_path = input_dir / "grid_split_assignments.csv"
    if split_path.exists() and not args.recompute_splits:
        split_df = pd.read_csv(split_path)
        if "split_role" in split_df.columns and "circuit_key" in split_df.columns:
            return split_df
    circuits = candidates[["circuit_key", "algorithm", "size", "circuit_kind"]].drop_duplicates().copy()
    if args.sizes:
        sizes = set(parse_csv_ints(args.sizes))
        circuits = circuits[circuits["size"].isin(sizes)].copy()
    return make_stratified_size_splits(
        circuits=circuits,
        test_fraction=args.test_fraction,
        repetitions=args.split_repetitions,
        seed=args.split_seed,
    )


def candidate_train_metrics_for_split(
    train_df: pd.DataFrame,
    target_tvd: float,
    risk_group: str,
    risk_aggregation: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (config_id, policy, scheme), group in train_df.groupby(["config_id", "policy", "scheme"], sort=False):
        risk, detail = compute_group_risk(group, risk_group, risk_aggregation)
        row = {
            "config_id": config_id,
            "policy": policy,
            "scheme": scheme,
            **summarize_metric_frame(group, "train", target_tvd),
            "train_selection_risk": risk,
            "train_selection_risk_detail_json": json.dumps(detail, sort_keys=True),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def select_configs(train_metrics: pd.DataFrame, target_tvd: float, objective: str) -> pd.DataFrame:
    selected_rows: List[Dict[str, object]] = []
    for (policy, scheme), group in train_metrics.groupby(["policy", "scheme"], sort=True):
        eligible = group[group["train_selection_risk"] <= target_tvd].copy()
        relaxed = False
        if eligible.empty:
            eligible = group.copy()
            relaxed = True
        if objective == "max_ssr":
            ascending = [False, True, True, True, True]
            sort_cols = ["train_median_ssr", "train_selection_risk", "train_q95_tvd", "train_median_tvd", "config_id"]
        elif objective == "max_parallel_ssr":
            ascending = [False, True, True, True, True]
            sort_cols = ["train_median_parallel_ssr", "train_selection_risk", "train_q95_tvd", "train_median_tvd", "config_id"]
        elif objective == "min_tvd":
            ascending = [True, True, False, True]
            sort_cols = ["train_median_tvd", "train_selection_risk", "train_median_ssr", "config_id"]
        else:
            raise ValueError(f"unsupported selection objective {objective!r}")
        chosen = eligible.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        chosen["selection_relaxed"] = bool(relaxed)
        selected_rows.append(chosen)
    return pd.DataFrame(selected_rows)


def run_selection(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cand_path = input_dir / "grid_candidate_circuit_level_metrics.csv"
    if not cand_path.exists():
        raise FileNotFoundError(f"missing {cand_path}; run/merge grid search first")
    candidates = pd.read_csv(cand_path)
    if args.policies:
        candidates = candidates[candidates["policy"].isin(parse_csv_strings(args.policies))].copy()
    if args.weighting_schemes:
        candidates = candidates[candidates["scheme"].isin(parse_csv_strings(args.weighting_schemes))].copy()
    if args.sizes:
        candidates = candidates[candidates["size"].isin(parse_csv_ints(args.sizes))].copy()
    if candidates.empty:
        raise RuntimeError("no candidate rows after filtering")

    split_df = load_or_make_splits(input_dir, candidates, args)
    split_df.to_csv(output_dir / "grid_size_robust_split_assignments.csv", index=False)

    train_metric_frames: List[pd.DataFrame] = []
    selected_frames: List[pd.DataFrame] = []
    selected_test_summary_rows: List[Dict[str, object]] = []
    selected_test_metric_frames: List[pd.DataFrame] = []

    for (split_id, rep), split_group in split_df.groupby(["split_id", "repetition"], sort=True):
        train_keys = set(split_group.loc[split_group["split_role"].isin(["train", "validation"]), "circuit_key"])
        test_keys = set(split_group.loc[split_group["split_role"] == "test", "circuit_key"])
        train_df = candidates[candidates["circuit_key"].isin(train_keys)].copy()
        test_df = candidates[candidates["circuit_key"].isin(test_keys)].copy()
        train_metrics = candidate_train_metrics_for_split(
            train_df=train_df,
            target_tvd=args.selection_target_tvd,
            risk_group=args.selection_risk_group,
            risk_aggregation=args.selection_risk_aggregation,
        )
        train_metrics.insert(0, "repetition", int(rep))
        train_metrics.insert(0, "split_mode", "stratified_size")
        train_metrics.insert(0, "split_id", split_id)
        train_metrics["selection_target_tvd"] = args.selection_target_tvd
        train_metrics["selection_risk_group"] = args.selection_risk_group
        train_metrics["selection_risk_aggregation"] = args.selection_risk_aggregation
        train_metrics["selection_objective"] = args.selection_objective
        train_metric_frames.append(train_metrics)

        selected = select_configs(train_metrics, args.selection_target_tvd, args.selection_objective)
        selected_frames.append(selected.copy())
        for _, selected_row in selected.iterrows():
            config_id = selected_row["config_id"]
            policy = selected_row["policy"]
            scheme = selected_row["scheme"]
            test_rows = test_df[
                (test_df["config_id"] == config_id)
                & (test_df["policy"] == policy)
                & (test_df["scheme"] == scheme)
            ].copy()
            test_risk, test_risk_detail = compute_group_risk(
                test_rows, args.selection_risk_group, args.selection_risk_aggregation
            )
            summary = {
                "split_id": split_id,
                "split_mode": "stratified_size",
                "repetition": int(rep),
                "policy": policy,
                "scheme": scheme,
                "selected_config_id": config_id,
                "selection_relaxed": bool(selected_row["selection_relaxed"]),
                "selection_target_tvd": args.selection_target_tvd,
                "selection_risk_group": args.selection_risk_group,
                "selection_risk_aggregation": args.selection_risk_aggregation,
                "selection_objective": args.selection_objective,
                "train_selection_risk": float(selected_row["train_selection_risk"]),
                "test_selection_risk": test_risk,
                "test_selection_risk_detail_json": json.dumps(test_risk_detail, sort_keys=True),
                **summarize_metric_frame(test_rows, "test", args.selection_target_tvd),
            }
            selected_test_summary_rows.append(summary)
            test_rows.insert(0, "selected_config_id", config_id)
            test_rows.insert(0, "repetition", int(rep))
            test_rows.insert(0, "split_id", split_id)
            selected_test_metric_frames.append(test_rows)

    train_metrics_all = pd.concat(train_metric_frames, ignore_index=True) if train_metric_frames else pd.DataFrame()
    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    selected_test_summary = pd.DataFrame(selected_test_summary_rows)
    selected_test_metrics = pd.concat(selected_test_metric_frames, ignore_index=True) if selected_test_metric_frames else pd.DataFrame()

    train_metrics_all.to_csv(output_dir / "grid_size_robust_candidate_train_metrics.csv", index=False)
    selected_all.to_csv(output_dir / "grid_size_robust_selected_configs.csv", index=False)
    selected_test_summary.to_csv(output_dir / "grid_size_robust_selected_test_summary.csv", index=False)
    selected_test_metrics.to_csv(output_dir / "grid_size_robust_selected_test_metrics.csv", index=False)

    if not selected_test_summary.empty:
        repeated = (
            selected_test_summary.groupby(["policy", "scheme"], sort=True)
            .agg(
                repetitions=("split_id", "nunique"),
                median_test_median_tvd=("test_median_tvd", "median"),
                q75_test_median_tvd=("test_median_tvd", lambda s: float(s.quantile(0.75))),
                median_test_q95_tvd=("test_q95_tvd", "median"),
                median_test_size_robust_risk=("test_selection_risk", "median"),
                median_test_max_tvd=("test_max_tvd", "median"),
                median_test_valid_rate_tvd_le_0p05=(f"test_{valid_col(args.selection_target_tvd)}", "median"),
                median_test_median_ssr=("test_median_ssr", "median"),
                median_test_median_parallel_ssr=("test_median_parallel_ssr", "median"),
                relaxed_selection_rate=("selection_relaxed", "mean"),
            )
            .reset_index()
        )
        repeated.to_csv(output_dir / "grid_size_robust_repeated_selected_summary.csv", index=False)
    if not selected_all.empty:
        freq = (
            selected_all.groupby(["policy", "scheme", "config_id"], sort=True)
            .agg(selections=("split_id", "count"), relaxed_rate=("selection_relaxed", "mean"))
            .reset_index()
        )
        totals = selected_all.groupby(["policy", "scheme"])["split_id"].count().reset_index(name="total")
        freq = freq.merge(totals, on=["policy", "scheme"], how="left")
        freq["selection_frequency"] = freq["selections"] / freq["total"]
        freq = freq.sort_values(["policy", "scheme", "selections"], ascending=[True, True, False])
        freq.to_csv(output_dir / "grid_size_robust_config_frequency.csv", index=False)

    pd.DataFrame([
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "candidate_rows": int(len(candidates)),
            "split_repetitions": int(args.split_repetitions),
            "test_fraction": float(args.test_fraction),
            "split_seed": int(args.split_seed),
            "selection_target_tvd": float(args.selection_target_tvd),
            "selection_risk_group": args.selection_risk_group,
            "selection_risk_aggregation": args.selection_risk_aggregation,
            "selection_objective": args.selection_objective,
        }
    ]).to_csv(output_dir / "grid_size_robust_selection_metadata.csv", index=False)

    print("Wrote size-robust selection results to:")
    print(f"  {output_dir.resolve()}")
    if (output_dir / "grid_size_robust_repeated_selected_summary.csv").exists():
        print("\nRepeated selected test summary:")
        print(pd.read_csv(output_dir / "grid_size_robust_repeated_selected_summary.csv").to_string(index=False))
    if (output_dir / "grid_size_robust_config_frequency.csv").exists():
        print("\nConfig frequency:")
        print(pd.read_csv(output_dir / "grid_size_robust_config_frequency.csv").head(40).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run size-aware selection on existing StableShots multi-QPU grid outputs")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--weighting-schemes", default=",".join(DEFAULT_SCHEMES))
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--split-mode", choices=["stratified_size"], default="stratified_size")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-repetitions", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--recompute-splits", action="store_true")
    parser.add_argument("--selection-target-tvd", type=float, default=0.05)
    parser.add_argument("--selection-risk-group", default="size")
    parser.add_argument(
        "--selection-risk-aggregation",
        choices=["max_group_q95", "mean_group_q95", "median_group_q95"],
        default="max_group_q95",
    )
    parser.add_argument("--selection-objective", choices=["max_ssr", "max_parallel_ssr", "min_tvd"], default="max_ssr")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.output_dir:
        args.output_dir = args.input_dir
    run_selection(args)


if __name__ == "__main__":
    main()
