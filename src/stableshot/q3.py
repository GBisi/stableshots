#!/usr/bin/env python3
"""
Analyze RQ3 scaled Hoeffding/Weissman factors with grouped calibration.

This script extends the original RQ3 scaled-bound analysis by comparing:

  1. global scaling:          one alpha for all validation traces
  2. size scaling:            one alpha per qubit size
  3. size_algorithm scaling:  one alpha per (size, algorithm) cell

It supports two input formats:

  A) trace_metrics.csv from the StableShots experiment script
     Required columns:
       trace_id, algorithm, size, backend, strategy, shots, tvd_to_reference
     Fixed-shot rows must have strategy names like fixed_5000 or fixed-5000.

  B) aggregate_fixed_subset_metrics.csv-style file
     Required columns:
       algorithm, size, fixed_budget, tvd
     Backend column can be backend, subset_key, or backend_labels.

The validation/test protocol matches the paper setup by default:
  --split-mode backend_holdout --test-fraction 0.2 --split-repetitions 100

For each split, scaling factors are learned on validation traces only and
applied to held-out test traces only.

Outputs
-------
  bound_trace_table.csv
  grouped_alpha_summary.csv
  grouped_scaled_bound_trace_metrics.csv
  grouped_scaled_bound_summary_by_split.csv
  grouped_scaled_bound_summary.csv
  grouped_scaled_bound_summary_by_group.csv
  split_assignments.csv

Budget cap behavior
-------------------
  The 20,000-shot prefix is the empirical reference. In the scaled-bound
  analysis, requests up to the largest non-reference budget, normally 18,000,
  are rounded up to the nearest available fixed budget. Requests above that
  largest non-reference budget are evaluated at 20,000 and marked as
  reference_cap_used=True. This includes requests above 20,000. Therefore,
  scaled-bound rows are not marked uncovered merely because they exceed 20,000;
  instead, cap dependence is exposed through reference_cap_rate and
  success_rate_before_reference_cap.

StableShots and Table-I-style comparison
--------------------------------------
  Optionally pass --stableshots-trace-metrics to compute plain StableShots
  held-out summaries on the same repeated test splits. Unlike the older
  sensitivity-only mode, the main StableShots row uses all held-out evaluations,
  including max-budget cases, exactly as in Table I. The output also reports
  stop_rate, i.e. the fraction of evaluations that stopped before the 20,000-
  shot reference cap. For scaled-bound rows, requests above 18,000 are always
  evaluated at 20,000 and stop_rate is the complementary non-cap rate.

Example
-------
  python analyze_rq3_grouped_scaling.py \
      --fixed-subset-metrics aggregate_fixed_subset_metrics.csv \
      --output-dir rq3_grouped_scaling \
      --taus 0.05 \
      --bound-delta 0.05 \
      --calibrations median,p75,p90 \
      --group-modes global,size,size_algorithm \
      --split-repetitions 100

Or, from trace_metrics.csv:

  python analyze_rq3_grouped_scaling.py \
      --trace-metrics trace_metrics.csv \
      --output-dir rq3_grouped_scaling
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_ALGORITHMS = ["dj", "qaoa", "qft", "qnn", "random", "vqe"]
DEFAULT_SIZES = [4, 6, 8, 10, 12, 14]
DEFAULT_BACKENDS = ["fake_fez", "fake_kyiv", "fake_marrakesh", "fake_sherbrooke", "fake_torino"]
DEFAULT_FIXED_BASELINES = [1000, 2500, 5000, 10000, 15000, 18000, 20000]
BOUND_SPECS = [
    ("hoeffding", "n_hoeffding", "alpha_hoeffding"),
    ("weissman", "n_weissman", "alpha_weissman"),
]
ALLOWED_CALIBRATIONS = {"median", "p75", "p90", "max"}
ALLOWED_GROUP_MODES = {"global", "size", "size_algorithm"}
ALLOWED_SPLIT_MODES = {"backend_holdout", "random"}


@dataclass(frozen=True)
class SplitDefinition:
    split_id: str
    split_mode: str
    repetition: int
    validation_trace_ids: tuple[str, ...]
    test_trace_ids: tuple[str, ...]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in parse_csv_strings(value)]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in parse_csv_strings(value)]


def format_float_for_name(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def normalize_backend_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "backend" in out.columns:
        out["backend"] = out["backend"].astype(str)
        return out
    if "subset_key" in out.columns:
        out["backend"] = out["subset_key"].astype(str)
        return out
    if "backend_labels" in out.columns:
        # aggregate CSVs sometimes store a singleton backend label or a list-like string.
        out["backend"] = (
            out["backend_labels"]
            .astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
            .str.replace("'", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.split(",")
            .str[0]
            .str.strip()
        )
        return out
    raise ValueError("input is missing a backend column: expected backend, subset_key, or backend_labels")


def build_trace_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["algorithm"] = out["algorithm"].astype(str)
    out["size"] = out["size"].astype(int)
    out["backend"] = out["backend"].astype(str)
    if "trace_id" not in out.columns:
        out["trace_id"] = out.apply(
            lambda r: f"{r['algorithm']}_{int(r['size'])}_{r['backend']}", axis=1
        )
    return out


def tvd_column_name_for_budget(budget: int) -> str:
    return f"tvd_{int(budget)}"


def fixed_budget_from_strategy(strategy: str) -> int | None:
    s = str(strategy)
    for prefix in ("fixed_", "fixed-"):
        if s.startswith(prefix):
            tail = s[len(prefix):]
            if tail.endswith("k"):
                return int(float(tail[:-1]) * 1000)
            return int(float(tail))
    return None


def load_fixed_wide_table(
    trace_metrics: Path | None,
    fixed_subset_metrics: Path | None,
    fixed_baselines: Sequence[int],
    algorithms: Sequence[str],
    sizes: Sequence[int],
    backends: Sequence[str],
) -> pd.DataFrame:
    if (trace_metrics is None) == (fixed_subset_metrics is None):
        raise ValueError("provide exactly one of --trace-metrics or --fixed-subset-metrics")

    if trace_metrics is not None:
        df = pd.read_csv(trace_metrics)
        required = {"algorithm", "size", "backend", "strategy", "shots", "tvd_to_reference"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"trace_metrics file is missing required columns: {sorted(missing)}")
        df = normalize_backend_column(df)
        df = build_trace_id_columns(df)
        df["fixed_budget"] = df["strategy"].apply(fixed_budget_from_strategy)
        fixed = df[df["fixed_budget"].notna()].copy()
        fixed["fixed_budget"] = fixed["fixed_budget"].astype(int)
        fixed["tvd"] = fixed["tvd_to_reference"].astype(float)
    else:
        df = pd.read_csv(fixed_subset_metrics)
        required = {"algorithm", "size", "fixed_budget", "tvd"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"fixed-subset file is missing required columns: {sorted(missing)}")
        fixed = normalize_backend_column(df)
        fixed = build_trace_id_columns(fixed)
        fixed["fixed_budget"] = fixed["fixed_budget"].astype(int)
        fixed["tvd"] = fixed["tvd"].astype(float)

    fixed = fixed[
        fixed["algorithm"].isin(algorithms)
        & fixed["size"].isin(sizes)
        & fixed["backend"].isin(backends)
        & fixed["fixed_budget"].isin(fixed_baselines)
    ].copy()

    if fixed.empty:
        raise ValueError("no fixed-budget rows remain after filtering")

    index_cols = ["trace_id", "algorithm", "size", "backend"]
    wide = fixed.pivot_table(
        index=index_cols,
        columns="fixed_budget",
        values="tvd",
        aggfunc="first",
    ).reset_index()
    wide.columns = [
        tvd_column_name_for_budget(c) if isinstance(c, (int, np.integer)) else c
        for c in wide.columns
    ]

    for budget in fixed_baselines:
        col = tvd_column_name_for_budget(int(budget))
        if col not in wide.columns:
            wide[col] = np.nan

    # The 20,000-shot prefix is the empirical reference. If it is not present
    # as an explicit fixed-budget row, add it with TVD=0 so cap-fallback
    # policies can be evaluated consistently.
    if 20000 in set(int(b) for b in fixed_baselines):
        wide[tvd_column_name_for_budget(20000)] = 0.0

    expected = len(algorithms) * len(sizes) * len(backends)
    if len(wide) != expected:
        print(
            f"WARNING: expected {expected} traces after filtering, found {len(wide)}. "
            "The script will continue with available traces."
        )

    return wide.sort_values(["algorithm", "size", "backend"]).reset_index(drop=True)


def empirical_sufficient_budget(row: pd.Series, tau: float, fixed_baselines: Sequence[int]) -> float:
    for budget in sorted(int(b) for b in fixed_baselines):
        value = row.get(tvd_column_name_for_budget(budget), np.nan)
        if pd.notna(value) and float(value) <= tau:
            return float(budget)
    return np.nan


def log_two_power_M_minus_2(M: int) -> float:
    if M <= 1:
        raise ValueError("M must be at least 2")
    if M < 50:
        return math.log((2**M) - 2)
    # Approximation avoids constructing huge integers for large alphabets.
    return M * math.log(2)


def hoeffding_shots_for_tvd(num_qubits: int, tau: float, delta: float) -> int:
    M = 2 ** int(num_qubits)
    n = (M**2 / (8 * tau**2)) * math.log((2 * M) / delta)
    return int(math.ceil(n))


def weissman_shots_for_tvd(num_qubits: int, tau: float, delta: float) -> int:
    M = 2 ** int(num_qubits)
    n = (log_two_power_M_minus_2(M) + math.log(1 / delta)) / (2 * tau**2)
    return int(math.ceil(n))


def add_bound_columns(
    wide: pd.DataFrame,
    fixed_baselines: Sequence[int],
    tau: float,
    delta: float,
) -> pd.DataFrame:
    out = wide.copy()
    out["bound_tau"] = float(tau)
    out["bound_delta"] = float(delta)
    out["n_empirical_tau"] = out.apply(
        lambda r: empirical_sufficient_budget(r, tau, fixed_baselines), axis=1
    )
    out["n_hoeffding"] = out["size"].apply(lambda q: hoeffding_shots_for_tvd(int(q), tau, delta))
    out["n_weissman"] = out["size"].apply(lambda q: weissman_shots_for_tvd(int(q), tau, delta))
    out["alpha_hoeffding"] = out["n_empirical_tau"] / out["n_hoeffding"]
    out["alpha_weissman"] = out["n_empirical_tau"] / out["n_weissman"]
    out["empirical_target_reached_by_fixed_budget"] = out["n_empirical_tau"].notna()
    return out


def choose_alpha(values: pd.Series, calibration: str) -> float:
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return np.nan
    if calibration == "median":
        return float(clean.median())
    if calibration == "p75":
        return float(clean.quantile(0.75))
    if calibration == "p90":
        return float(clean.quantile(0.90))
    if calibration == "max":
        return float(clean.max())
    raise ValueError(f"unsupported calibration: {calibration}")


def reference_cap_budget(fixed_baselines: Sequence[int]) -> int:
    return 20000 if 20000 in set(int(b) for b in fixed_baselines) else max(int(b) for b in fixed_baselines)


def max_non_reference_budget(fixed_baselines: Sequence[int]) -> int:
    budgets = sorted(int(b) for b in fixed_baselines if int(b) < 20000)
    if not budgets:
        raise ValueError("fixed_baselines must include at least one budget below 20,000")
    return budgets[-1]


def round_up_to_available_budget(
    value: float,
    fixed_baselines: Sequence[int],
    reference_cap_lower_threshold: int = 19000,
) -> float:
    """Round a scaled request under the reference-cap fallback rule.

    Rules:
      * Requests up to the largest non-reference budget, normally 18,000, are
        rounded up to the nearest available non-reference fixed budget.
      * Requests above the largest non-reference budget are evaluated at the
        reference cap, normally 20,000, and marked elsewhere as reference-cap
        uses. This includes requests above 20,000.
      * NaN requests remain uncovered.

    The reference_cap_lower_threshold argument is accepted for backward
    compatibility with older runs, but is not used by this rule.
    """
    if pd.isna(value):
        return np.nan

    cap = reference_cap_budget(fixed_baselines)
    max_regular = max_non_reference_budget(fixed_baselines)

    if value <= max_regular:
        for budget in sorted(int(b) for b in fixed_baselines if int(b) < cap):
            if value <= budget:
                return float(budget)

    return float(cap)


def tvd_at_budget(row: pd.Series, budget: float) -> float:
    if pd.isna(budget):
        return np.nan
    return float(row.get(tvd_column_name_for_budget(int(budget)), np.nan))


def make_split_id(mode: str, repetition: int) -> str:
    return f"{mode}_rep{repetition:03d}"


def make_splits(
    wide: pd.DataFrame,
    mode: str,
    test_fraction: float,
    repetitions: int,
    seed: int,
) -> list[SplitDefinition]:
    if mode not in ALLOWED_SPLIT_MODES:
        raise ValueError(f"unsupported split mode: {mode}")
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be in (0, 1)")
    if repetitions <= 0:
        raise ValueError("split_repetitions must be positive")

    trace_ids = sorted(wide["trace_id"].unique())
    splits: list[SplitDefinition] = []

    if mode == "random":
        n_test = max(1, min(len(trace_ids) - 1, int(round(len(trace_ids) * test_fraction))))
        for rep in range(repetitions):
            rng = np.random.default_rng(seed + rep)
            permuted = list(rng.permutation(trace_ids))
            test_ids = tuple(sorted(str(x) for x in permuted[:n_test]))
            validation_ids = tuple(sorted(set(trace_ids) - set(test_ids)))
            splits.append(SplitDefinition(make_split_id(mode, rep), mode, rep, validation_ids, test_ids))
        return splits

    # backend_holdout: hold out one or more backend traces per algorithm-size cell.
    cells = {}
    for (algorithm, size), group in wide.groupby(["algorithm", "size"], sort=True):
        cells[(str(algorithm), int(size))] = group.sort_values("backend")

    for rep in range(repetitions):
        rng = np.random.default_rng(seed + rep)
        test_ids_set: set[str] = set()
        for key in sorted(cells):
            cell = cells[key]
            if len(cell) < 2:
                raise ValueError(f"cell {key} has fewer than two backend traces")
            n_test = max(1, min(len(cell) - 1, int(round(len(cell) * test_fraction))))
            indices = rng.choice(len(cell), size=n_test, replace=False)
            for idx in indices:
                test_ids_set.add(str(cell.iloc[int(idx)]["trace_id"]))
        test_ids = tuple(sorted(test_ids_set))
        validation_ids = tuple(sorted(set(trace_ids) - set(test_ids)))
        splits.append(SplitDefinition(make_split_id(mode, rep), mode, rep, validation_ids, test_ids))

    return splits


def build_split_assignments(wide: pd.DataFrame, splits: Sequence[SplitDefinition]) -> pd.DataFrame:
    meta = wide[["trace_id", "algorithm", "size", "backend"]].drop_duplicates()
    meta_by_id = {str(r.trace_id): r for r in meta.itertuples(index=False)}
    rows = []
    for split in splits:
        for role, ids in (("validation", split.validation_trace_ids), ("test", split.test_trace_ids)):
            for trace_id in ids:
                r = meta_by_id[str(trace_id)]
                rows.append(
                    {
                        "split_id": split.split_id,
                        "split_mode": split.split_mode,
                        "repetition": split.repetition,
                        "split_role": role,
                        "trace_id": trace_id,
                        "algorithm": r.algorithm,
                        "size": int(r.size),
                        "backend": r.backend,
                    }
                )
    return pd.DataFrame(rows)


def grouping_columns(group_mode: str) -> list[str]:
    if group_mode == "global":
        return []
    if group_mode == "size":
        return ["size"]
    if group_mode == "size_algorithm":
        return ["size", "algorithm"]
    raise ValueError(f"unsupported group mode: {group_mode}")


def group_key_dict(row: pd.Series, group_cols: Sequence[str]) -> dict[str, object]:
    return {col: row[col] for col in group_cols}


def group_label_from_values(values: Sequence[object]) -> str:
    if not values:
        return "global"
    return "|".join(str(v) for v in values)


def compute_alpha_table(
    validation: pd.DataFrame,
    alpha_col: str,
    group_mode: str,
    calibration: str,
) -> pd.DataFrame:
    group_cols = grouping_columns(group_mode)
    rows = []

    if not group_cols:
        alpha = choose_alpha(validation[alpha_col], calibration)
        values = validation[alpha_col].dropna().astype(float)
        rows.append(
            {
                "group_mode": group_mode,
                "group_label": "global",
                "alpha": alpha,
                "calibration_trace_count": int(values.count()),
                "alpha_min": float(values.min()) if not values.empty else np.nan,
                "alpha_median": float(values.median()) if not values.empty else np.nan,
                "alpha_p75": float(values.quantile(0.75)) if not values.empty else np.nan,
                "alpha_p90": float(values.quantile(0.90)) if not values.empty else np.nan,
                "alpha_max": float(values.max()) if not values.empty else np.nan,
            }
        )
        return pd.DataFrame(rows)

    for key, group in validation.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        values = group[alpha_col].dropna().astype(float)
        row = {
            "group_mode": group_mode,
            "group_label": group_label_from_values(key),
            "alpha": choose_alpha(values, calibration),
            "calibration_trace_count": int(values.count()),
            "alpha_min": float(values.min()) if not values.empty else np.nan,
            "alpha_median": float(values.median()) if not values.empty else np.nan,
            "alpha_p75": float(values.quantile(0.75)) if not values.empty else np.nan,
            "alpha_p90": float(values.quantile(0.90)) if not values.empty else np.nan,
            "alpha_max": float(values.max()) if not values.empty else np.nan,
        }
        for col, value in zip(group_cols, key):
            row[col] = value
        rows.append(row)

    return pd.DataFrame(rows)


def attach_group_alpha(
    test: pd.DataFrame,
    alpha_table: pd.DataFrame,
    group_mode: str,
    fallback_alpha: float,
) -> pd.DataFrame:
    out = test.copy()
    group_cols = grouping_columns(group_mode)

    if not group_cols:
        out["alpha_train"] = float(alpha_table.iloc[0]["alpha"]) if not alpha_table.empty else np.nan
        out["alpha_group_label"] = "global"
        out["alpha_calibration_trace_count"] = int(alpha_table.iloc[0]["calibration_trace_count"]) if not alpha_table.empty else 0
        out["alpha_fallback_used"] = False
        return out

    merge_cols = list(group_cols)
    keep_cols = merge_cols + ["group_label", "alpha", "calibration_trace_count"]
    merged = out.merge(alpha_table[keep_cols], on=merge_cols, how="left")
    merged["alpha_fallback_used"] = merged["alpha"].isna()
    merged["alpha_train"] = merged["alpha"].fillna(fallback_alpha)
    merged["alpha_group_label"] = merged["group_label"].fillna("fallback_global")
    merged["alpha_calibration_trace_count"] = merged["calibration_trace_count"].fillna(0).astype(int)
    merged = merged.drop(columns=["alpha", "group_label", "calibration_trace_count"])
    return merged


def evaluate_grouped_scaled_policy(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    fixed_baselines: Sequence[int],
    bound: str,
    bound_col: str,
    alpha_col: str,
    group_mode: str,
    calibration: str,
    tau: float,
    delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha_table = compute_alpha_table(validation, alpha_col, group_mode, calibration)
    fallback_alpha = choose_alpha(validation[alpha_col], calibration)
    detail = attach_group_alpha(test, alpha_table, group_mode, fallback_alpha)

    detail["n_scaled_raw"] = np.ceil(detail["alpha_train"] * detail[bound_col])
    detail["reference_cap_budget"] = reference_cap_budget(fixed_baselines)
    detail["max_non_reference_budget"] = max_non_reference_budget(fixed_baselines)
    detail["reference_cap_lower_threshold"] = 19000
    detail["n_scaled_budget"] = detail["n_scaled_raw"].apply(
        lambda n: round_up_to_available_budget(
            n,
            fixed_baselines,
            reference_cap_lower_threshold=19000,
        )
    )
    detail["above_regular_budget_requested"] = detail["n_scaled_raw"] > detail["max_non_reference_budget"]
    detail["near_reference_cap_requested"] = (
        (detail["n_scaled_raw"] > detail["max_non_reference_budget"])
        & (detail["n_scaled_raw"] < detail["reference_cap_budget"])
    )
    detail["gap_uncovered_requested"] = False
    detail["above_or_equal_reference_cap_requested"] = detail["n_scaled_raw"] >= detail["reference_cap_budget"]
    detail["reference_cap_used"] = detail["n_scaled_budget"] == detail["reference_cap_budget"]
    detail["tvd_scaled"] = detail.apply(lambda r: tvd_at_budget(r, r["n_scaled_budget"]), axis=1)
    detail["scaled_uncovered"] = detail["n_scaled_budget"].isna()
    detail["scaled_success"] = (detail["tvd_scaled"] <= tau).fillna(False)
    detail["scaled_success_before_reference_cap"] = (
        detail["scaled_success"] & ~detail["reference_cap_used"]
    )

    detail["bound"] = bound
    detail["bound_col"] = bound_col
    detail["alpha_col"] = alpha_col
    detail["group_mode"] = group_mode
    detail["calibration"] = calibration
    detail["bound_tau"] = tau
    detail["bound_delta"] = delta

    return detail, alpha_table


def summarize_detail_by_split(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["split_id", "split_mode", "repetition", "bound_tau", "bound_delta", "bound", "group_mode", "calibration"]
    for key, group in detail.groupby(group_cols, dropna=False, sort=True):
        data = dict(zip(group_cols, key))
        evaluable = group[~group["scaled_uncovered"]]
        data.update(
            {
                "test_trace_count": int(len(group)),
                "success_rate_all": float(group["scaled_success"].mean()) if len(group) else np.nan,
                "uncovered_rate": float(group["scaled_uncovered"].mean()) if len(group) else np.nan,
                "success_rate_evaluable": float(evaluable["scaled_success"].mean()) if len(evaluable) else np.nan,
                "success_rate_before_reference_cap": float(group["scaled_success_before_reference_cap"].mean()) if len(group) else np.nan,
                "above_regular_budget_requested_rate": float(group["above_regular_budget_requested"].mean()) if len(group) else np.nan,
                "near_reference_cap_requested_rate": float(group["near_reference_cap_requested"].mean()) if len(group) else np.nan,
                "gap_uncovered_requested_rate": float(group["gap_uncovered_requested"].mean()) if len(group) else np.nan,
                "above_or_equal_reference_cap_requested_rate": float(group["above_or_equal_reference_cap_requested"].mean()) if len(group) else np.nan,
                "reference_cap_rate": float(group["reference_cap_used"].mean()) if len(group) else np.nan,
                "median_scaled_budget": float(group["n_scaled_budget"].median()),
                "mean_scaled_budget": float(group["n_scaled_budget"].mean()),
                "median_scaled_tvd": float(group["tvd_scaled"].median()),
                "mean_scaled_tvd": float(group["tvd_scaled"].mean()),
                "fallback_rate": float(group["alpha_fallback_used"].mean()) if "alpha_fallback_used" in group else 0.0,
            }
        )
        rows.append(data)
    return pd.DataFrame(rows)


def summarize_detail_overall(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["bound_tau", "bound_delta", "bound", "group_mode", "calibration"]
    for key, group in detail.groupby(group_cols, dropna=False, sort=True):
        data = dict(zip(group_cols, key))
        evaluable = group[~group["scaled_uncovered"]]
        data.update(
            {
                "test_eval_count": int(len(group)),
                "unique_test_traces": int(group["trace_id"].nunique()),
                "success_rate_all": float(group["scaled_success"].mean()) if len(group) else np.nan,
                "uncovered_rate": float(group["scaled_uncovered"].mean()) if len(group) else np.nan,
                "success_rate_evaluable": float(evaluable["scaled_success"].mean()) if len(evaluable) else np.nan,
                "success_rate_before_reference_cap": float(group["scaled_success_before_reference_cap"].mean()) if len(group) else np.nan,
                "above_regular_budget_requested_rate": float(group["above_regular_budget_requested"].mean()) if len(group) else np.nan,
                "near_reference_cap_requested_rate": float(group["near_reference_cap_requested"].mean()) if len(group) else np.nan,
                "gap_uncovered_requested_rate": float(group["gap_uncovered_requested"].mean()) if len(group) else np.nan,
                "above_or_equal_reference_cap_requested_rate": float(group["above_or_equal_reference_cap_requested"].mean()) if len(group) else np.nan,
                "reference_cap_rate": float(group["reference_cap_used"].mean()) if len(group) else np.nan,
                "median_scaled_budget": float(group["n_scaled_budget"].median()),
                "mean_scaled_budget": float(group["n_scaled_budget"].mean()),
                "p90_scaled_budget": float(group["n_scaled_budget"].quantile(0.90)),
                "median_scaled_tvd": float(group["tvd_scaled"].median()),
                "mean_scaled_tvd": float(group["tvd_scaled"].mean()),
                "p90_scaled_tvd": float(group["tvd_scaled"].quantile(0.90)),
                "max_scaled_tvd": float(group["tvd_scaled"].max()),
                "fallback_rate": float(group["alpha_fallback_used"].mean()) if "alpha_fallback_used" in group else 0.0,
            }
        )
        rows.append(data)
    return pd.DataFrame(rows)


def summarize_detail_by_group(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_cols = ["bound_tau", "bound_delta", "bound", "group_mode", "calibration"]
    for group_col in ["size", "algorithm", "backend"]:
        cols = base_cols + [group_col]
        for key, group in detail.groupby(cols, dropna=False, sort=True):
            data = dict(zip(cols, key))
            data["reported_group_col"] = group_col
            data["reported_group_value"] = data.pop(group_col)
            evaluable = group[~group["scaled_uncovered"]]
            data.update(
                {
                    "test_eval_count": int(len(group)),
                    "unique_test_traces": int(group["trace_id"].nunique()),
                    "success_rate_all": float(group["scaled_success"].mean()) if len(group) else np.nan,
                    "uncovered_rate": float(group["scaled_uncovered"].mean()) if len(group) else np.nan,
                    "success_rate_evaluable": float(evaluable["scaled_success"].mean()) if len(evaluable) else np.nan,
                    "success_rate_before_reference_cap": float(group["scaled_success_before_reference_cap"].mean()) if len(group) else np.nan,
                    "above_regular_budget_requested_rate": float(group["above_regular_budget_requested"].mean()) if len(group) else np.nan,
                    "near_reference_cap_requested_rate": float(group["near_reference_cap_requested"].mean()) if len(group) else np.nan,
                    "gap_uncovered_requested_rate": float(group["gap_uncovered_requested"].mean()) if len(group) else np.nan,
                    "above_or_equal_reference_cap_requested_rate": float(group["above_or_equal_reference_cap_requested"].mean()) if len(group) else np.nan,
                    "reference_cap_rate": float(group["reference_cap_used"].mean()) if len(group) else np.nan,
                    "median_scaled_budget": float(group["n_scaled_budget"].median()),
                    "mean_scaled_budget": float(group["n_scaled_budget"].mean()),
                    "median_scaled_tvd": float(group["tvd_scaled"].median()),
                    "max_scaled_tvd": float(group["tvd_scaled"].max()),
                }
            )
            rows.append(data)
    return pd.DataFrame(rows)


def summarize_alpha_tables(alpha_tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not alpha_tables:
        return pd.DataFrame()
    return pd.concat(alpha_tables, ignore_index=True)



def resolve_existing_path(path: Path, description: str) -> Path:
    """Return an existing path, or raise an actionable FileNotFoundError."""
    if path.exists():
        return path
    parent = path.parent if str(path.parent) not in {"", "."} else Path.cwd()
    candidates = []
    if parent.exists() and parent.is_dir():
        names = [
            "aggregate_stableshots_subset_metrics.csv",
            "aggregate_policy_subset_metrics.csv",
            "materialized_trace_metrics.csv",
            "trace_metrics.csv",
            "validation_test_selected_trace_metrics.csv",
            "grid_selected_test_metrics.csv",
        ]
        for name in names:
            cand = parent / name
            if cand.exists():
                candidates.append(str(cand))
    hint = ""
    if candidates:
        hint = "\nExisting candidate files in the same directory:\n  " + "\n  ".join(candidates)
    raise FileNotFoundError(f"{description} not found: {path}{hint}")


def load_stableshots_trace_metrics(
    path: Path,
    wide: pd.DataFrame,
    splits: Sequence[SplitDefinition],
    strategy_filter: str,
    scheme_filter: str = "uniform",
    k_filter: int = 1,
    policy_filter: str = "local_only_offline_aggregate",
) -> pd.DataFrame:
    """Load StableShots rows and return held-out test evaluations.

    Supported inputs include:
      * validation_test_selected_trace_metrics.csv: already split/test rows;
      * trace_metrics.csv or materialized_trace_metrics.csv: one row per trace/strategy;
      * aggregate_stableshots_subset_metrics.csv: multi-QPU local-only aggregate output;
      * aggregate_policy_subset_metrics.csv: multi-QPU policy output.

    For multi-QPU aggregate files, the default sensitivity comparison uses the
    singleton-backend rows only: k=1, scheme=uniform, and policy=
    local_only_offline_aggregate when that column exists. This matches the
    single-trace held-out split used by the bound analysis.
    """
    path = resolve_existing_path(path, "StableShots trace metrics file")
    df = pd.read_csv(path)

    # Optional filters for multi-QPU aggregate outputs.
    if "k" in df.columns and k_filter is not None:
        df = df[df["k"].astype(int).eq(int(k_filter))].copy()
    if "scheme" in df.columns and scheme_filter:
        df = df[df["scheme"].astype(str).eq(str(scheme_filter))].copy()
    if "policy" in df.columns and policy_filter:
        df = df[df["policy"].astype(str).eq(str(policy_filter))].copy()

    df = normalize_backend_column(df)
    df = build_trace_id_columns(df)

    # Normalize metric columns across the supported formats.
    if "tvd_to_reference" not in df.columns and "tvd" in df.columns:
        df["tvd_to_reference"] = df["tvd"].astype(float)
    if "shots" not in df.columns:
        if "adaptive_shots_total" in df.columns:
            df["shots"] = df["adaptive_shots_total"].astype(float)
        elif "max_backend_shots" in df.columns:
            df["shots"] = df["max_backend_shots"].astype(float)
    if "shots" not in df.columns:
        raise ValueError("StableShots file must contain shots, adaptive_shots_total, or max_backend_shots")
    if "tvd_to_reference" not in df.columns:
        raise ValueError("StableShots file must contain tvd_to_reference or tvd")

    if "split_role" in df.columns:
        df = df[df["split_role"].astype(str).str.lower().eq("test")].copy()

    # If the file already carries split identifiers, assume it is already a
    # held-out validation/test output and keep its rows.
    if {"split_id", "repetition"}.issubset(df.columns):
        out = df.copy()
        if "split_mode" not in out.columns:
            out["split_mode"] = "unknown"
        return out

    # Otherwise interpret it as ordinary trace-level output and keep one
    # StableShots strategy/config. If no strategy is provided and multiple
    # configs exist, fail loudly to avoid accidental averaging over configs.
    if strategy_filter:
        mask = pd.Series(False, index=df.index)
        if "strategy" in df.columns:
            mask = mask | df["strategy"].astype(str).eq(strategy_filter)
        if "config_id" in df.columns:
            mask = mask | df["config_id"].astype(str).eq(strategy_filter)
        df = df[mask].copy()
    else:
        if "strategy_family" in df.columns:
            df = df[df["strategy_family"].astype(str).eq("stable_shots")].copy()
        elif "strategy" in df.columns:
            # Multi-QPU aggregate outputs may not have strategy_family; keep rows
            # that look like StableShots if a strategy column is present.
            stable_mask = df["strategy"].astype(str).str.contains("stable", case=False, na=False)
            if stable_mask.any():
                df = df[stable_mask].copy()
        strategies = sorted(df["strategy"].dropna().astype(str).unique()) if "strategy" in df.columns else []
        config_ids = sorted(df["config_id"].dropna().astype(str).unique()) if "config_id" in df.columns else []
        if len(strategies) > 1 or len(config_ids) > 1:
            raise ValueError(
                "StableShots trace file contains multiple strategies/configs; pass --stableshots-strategy "
                "with the selected strategy or config_id."
            )

    if df.empty:
        raise ValueError(
            "no StableShots rows found after filtering. Check --stableshots-strategy, "
            "--stableshots-scheme, --stableshots-k, and --stableshots-policy."
        )

    st_by_trace = df.set_index("trace_id", drop=False)
    rows = []
    for split in splits:
        for trace_id in split.test_trace_ids:
            if trace_id not in st_by_trace.index:
                continue
            row = st_by_trace.loc[trace_id]
            if isinstance(row, pd.DataFrame):
                # Duplicate singleton rows can happen if an aggregate output still
                # contains equivalent weighting schemes. After filtering by scheme,
                # use the first remaining row deterministically.
                row = row.iloc[0]
            out = row.to_dict()
            out["split_id"] = split.split_id
            out["split_mode"] = split.split_mode
            out["repetition"] = split.repetition
            out["split_role"] = "test"
            rows.append(out)
    if not rows:
        raise ValueError(
            "no StableShots rows matched generated test splits. For multi-QPU outputs, "
            "make sure you passed aggregate_stableshots_subset_metrics.csv or "
            "aggregate_policy_subset_metrics.csv and that --stableshots-k/--stableshots-scheme match."
        )
    return pd.DataFrame(rows)


def table_threshold_columns(prefix: str, tvd: pd.Series, thresholds: Sequence[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for threshold in thresholds:
        name = f"{prefix}le_{format_float_for_name(float(threshold))}_rate"
        out[name] = float((tvd.astype(float) <= float(threshold)).mean()) if len(tvd) else np.nan
    return out


def infer_stableshots_stop_flags(detail: pd.DataFrame, reference_cap: float = 20000.0) -> pd.Series:
    """Infer whether StableShots stopped before the max/reference cap.

    Preference order:
      1. explicit stop_reason, if present;
      2. per_backend_stop_reason JSON, if present;
      3. shots < reference_cap fallback.
    """
    if "stop_reason" in detail.columns:
        reasons = detail["stop_reason"].fillna("").astype(str).str.lower()
        explicit = reasons.ne("")
        stopped = ~(reasons.str.contains("max_budget") | reasons.str.contains("cap"))
        return stopped.where(explicit, detail["shots"].astype(float) < float(reference_cap))

    if "per_backend_stop_reason" in detail.columns:
        flags = []
        for raw, shots in zip(detail["per_backend_stop_reason"], detail["shots"]):
            parsed = None
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None
            if isinstance(parsed, dict) and parsed:
                values = [str(v).lower() for v in parsed.values()]
                flags.append(not all(("max_budget" in v or "cap" in v) for v in values))
            else:
                flags.append(float(shots) < float(reference_cap))
        return pd.Series(flags, index=detail.index, dtype=bool)

    return detail["shots"].astype(float) < float(reference_cap)


def summarize_stableshots_sensitivity(
    stableshots: pd.DataFrame,
    taus: Sequence[float],
    covered_threshold: float,
    table_thresholds: Sequence[float] = (0.01, 0.05, 0.10),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return StableShots trace details plus all-trace and covered-only summaries.

    The all-trace summary is the one to compare with Table I: max-budget rows
    remain included, and stop_rate reports how often the stability rule stopped
    before the reference cap. The covered-only columns are retained for optional
    sensitivity analysis but are not the main comparison requested by Table I.
    """
    detail = stableshots.copy()
    detail["shots"] = detail["shots"].astype(float)
    detail["tvd_to_reference"] = detail["tvd_to_reference"].astype(float)
    detail["stableshots_excluded_over_threshold"] = detail["shots"] > float(covered_threshold)
    detail["stableshots_reference_cap_used"] = detail["shots"] >= 20000
    detail["stableshots_stopped"] = infer_stableshots_stop_flags(detail, reference_cap=20000.0)

    overall_rows = []
    group_rows = []
    table_rows = []
    for tau in taus:
        detail[f"stableshots_success_le_{format_float_for_name(tau)}"] = detail["tvd_to_reference"] <= tau
        covered = detail[~detail["stableshots_excluded_over_threshold"]]
        common = {
            "method": "stableshots",
            "bound_tau": tau,
            "covered_threshold": covered_threshold,
            "test_eval_count": int(len(detail)),
            "unique_test_traces": int(detail["trace_id"].nunique()),
            "success_rate_all": float((detail["tvd_to_reference"] <= tau).mean()) if len(detail) else np.nan,
            "median_shots": float(detail["shots"].median()) if len(detail) else np.nan,
            "mean_shots": float(detail["shots"].mean()) if len(detail) else np.nan,
            "median_tvd": float(detail["tvd_to_reference"].median()) if len(detail) else np.nan,
            "mean_tvd": float(detail["tvd_to_reference"].mean()) if len(detail) else np.nan,
            "max_tvd": float(detail["tvd_to_reference"].max()) if len(detail) else np.nan,
            "stop_rate": float(detail["stableshots_stopped"].mean()) if len(detail) else np.nan,
            "reference_cap_rate": float(detail["stableshots_reference_cap_used"].mean()) if len(detail) else np.nan,
            "excluded_over_threshold_rate": float(detail["stableshots_excluded_over_threshold"].mean()) if len(detail) else np.nan,
        }
        common.update(table_threshold_columns("", detail["tvd_to_reference"], table_thresholds))
        table_rows.append(common)
        overall_row = dict(common)
        overall_row.update(
            {
                "covered_eval_count": int(len(covered)),
                "covered_success_rate": float((covered["tvd_to_reference"] <= tau).mean()) if len(covered) else np.nan,
                "covered_median_shots": float(covered["shots"].median()) if len(covered) else np.nan,
                "covered_mean_shots": float(covered["shots"].mean()) if len(covered) else np.nan,
                "covered_median_tvd": float(covered["tvd_to_reference"].median()) if len(covered) else np.nan,
                "covered_mean_tvd": float(covered["tvd_to_reference"].mean()) if len(covered) else np.nan,
                "covered_max_tvd": float(covered["tvd_to_reference"].max()) if len(covered) else np.nan,
            }
        )
        overall_rows.append(overall_row)

        for group_col in ["size", "algorithm", "backend"]:
            for group_value, group in detail.groupby(group_col, dropna=False, sort=True):
                covered_g = group[~group["stableshots_excluded_over_threshold"]]
                row = {
                    "bound_tau": tau,
                    "covered_threshold": covered_threshold,
                    "reported_group_col": group_col,
                    "reported_group_value": group_value,
                    "test_eval_count": int(len(group)),
                    "unique_test_traces": int(group["trace_id"].nunique()),
                    "excluded_over_threshold_rate": float(group["stableshots_excluded_over_threshold"].mean()) if len(group) else np.nan,
                    "reference_cap_rate": float(group["stableshots_reference_cap_used"].mean()) if len(group) else np.nan,
                    "stop_rate": float(group["stableshots_stopped"].mean()) if len(group) else np.nan,
                    "success_rate_all": float((group["tvd_to_reference"] <= tau).mean()) if len(group) else np.nan,
                    "median_shots": float(group["shots"].median()) if len(group) else np.nan,
                    "mean_shots": float(group["shots"].mean()) if len(group) else np.nan,
                    "median_tvd": float(group["tvd_to_reference"].median()) if len(group) else np.nan,
                    "max_tvd": float(group["tvd_to_reference"].max()) if len(group) else np.nan,
                    "covered_eval_count": int(len(covered_g)),
                    "covered_success_rate": float((covered_g["tvd_to_reference"] <= tau).mean()) if len(covered_g) else np.nan,
                    "covered_median_shots": float(covered_g["shots"].median()) if len(covered_g) else np.nan,
                    "covered_mean_shots": float(covered_g["shots"].mean()) if len(covered_g) else np.nan,
                    "covered_max_tvd": float(covered_g["tvd_to_reference"].max()) if len(covered_g) else np.nan,
                }
                row.update(table_threshold_columns("", group["tvd_to_reference"], table_thresholds))
                group_rows.append(row)
    return detail, pd.DataFrame(overall_rows), pd.DataFrame(group_rows), pd.DataFrame(table_rows)


def summarize_scaled_table1_comparison(
    detail: pd.DataFrame,
    table_thresholds: Sequence[float],
) -> pd.DataFrame:
    """Table-I-style summaries for scaled-bound policies.

    For scaled bounds, stop_rate means the policy stayed within the regular
    non-reference budget grid (<=18k by default). reference_cap_rate reports the
    fraction defaulted to 20k because the scaled request exceeded 18k.
    """
    if detail.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["bound_tau", "bound_delta", "bound", "group_mode", "calibration"]
    for key, group in detail.groupby(group_cols, dropna=False, sort=True):
        data = dict(zip(group_cols, key))
        tvd = group["tvd_scaled"].astype(float)
        row = {
            "method": "scaled_bound",
            **data,
            "test_eval_count": int(len(group)),
            "unique_test_traces": int(group["trace_id"].nunique()),
            "success_rate_all": float(group["scaled_success"].mean()) if len(group) else np.nan,
            "median_shots": float(group["n_scaled_budget"].median()) if len(group) else np.nan,
            "mean_shots": float(group["n_scaled_budget"].mean()) if len(group) else np.nan,
            "median_tvd": float(tvd.median()) if len(group) else np.nan,
            "mean_tvd": float(tvd.mean()) if len(group) else np.nan,
            "max_tvd": float(tvd.max()) if len(group) else np.nan,
            "stop_rate": float((~group["reference_cap_used"].astype(bool)).mean()) if len(group) else np.nan,
            "reference_cap_rate": float(group["reference_cap_used"].mean()) if len(group) else np.nan,
            "uncovered_rate": float(group["scaled_uncovered"].mean()) if len(group) else np.nan,
            "above_regular_budget_requested_rate": float(group["above_regular_budget_requested"].mean()) if len(group) else np.nan,
        }
        row.update(table_threshold_columns("", tvd, table_thresholds))
        rows.append(row)
    return pd.DataFrame(rows)

def run_analysis(args: argparse.Namespace) -> None:
    fixed_baselines = parse_csv_ints(args.fixed_baselines)
    algorithms = parse_csv_strings(args.algorithms)
    sizes = parse_csv_ints(args.sizes)
    backends = parse_csv_strings(args.backends)
    taus = parse_csv_floats(args.taus)
    calibrations = parse_csv_strings(args.calibrations)
    group_modes = parse_csv_strings(args.group_modes)
    table_thresholds = parse_csv_floats(args.table_thresholds)

    unknown_cal = sorted(set(calibrations) - ALLOWED_CALIBRATIONS)
    if unknown_cal:
        raise ValueError(f"unsupported calibrations: {unknown_cal}; allowed: {sorted(ALLOWED_CALIBRATIONS)}")
    unknown_modes = sorted(set(group_modes) - ALLOWED_GROUP_MODES)
    if unknown_modes:
        raise ValueError(f"unsupported group modes: {unknown_modes}; allowed: {sorted(ALLOWED_GROUP_MODES)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_metrics = Path(args.trace_metrics) if args.trace_metrics else None
    fixed_subset_metrics = Path(args.fixed_subset_metrics) if args.fixed_subset_metrics else None
    wide = load_fixed_wide_table(
        trace_metrics=trace_metrics,
        fixed_subset_metrics=fixed_subset_metrics,
        fixed_baselines=fixed_baselines,
        algorithms=algorithms,
        sizes=sizes,
        backends=backends,
    )

    splits = make_splits(
        wide=wide,
        mode=args.split_mode,
        test_fraction=args.test_fraction,
        repetitions=args.split_repetitions,
        seed=args.split_seed,
    )
    split_assignments = build_split_assignments(wide, splits)

    all_trace_tables = []
    all_detail = []
    all_alpha_tables = []

    for tau in taus:
        scaled = add_bound_columns(wide, fixed_baselines, tau=tau, delta=args.bound_delta)
        all_trace_tables.append(scaled)
        by_id = {str(r.trace_id): i for i, r in scaled.iterrows()}

        for split in splits:
            validation_idx = [by_id[trace_id] for trace_id in split.validation_trace_ids if trace_id in by_id]
            test_idx = [by_id[trace_id] for trace_id in split.test_trace_ids if trace_id in by_id]
            validation = scaled.loc[validation_idx].copy()
            test = scaled.loc[test_idx].copy()

            for bound, bound_col, alpha_col in BOUND_SPECS:
                for calibration in calibrations:
                    for group_mode in group_modes:
                        detail, alpha_table = evaluate_grouped_scaled_policy(
                            validation=validation,
                            test=test,
                            fixed_baselines=fixed_baselines,
                            bound=bound,
                            bound_col=bound_col,
                            alpha_col=alpha_col,
                            group_mode=group_mode,
                            calibration=calibration,
                            tau=tau,
                            delta=args.bound_delta,
                        )
                        detail.insert(0, "split_id", split.split_id)
                        detail.insert(1, "split_mode", split.split_mode)
                        detail.insert(2, "repetition", split.repetition)

                        alpha_table.insert(0, "split_id", split.split_id)
                        alpha_table.insert(1, "split_mode", split.split_mode)
                        alpha_table.insert(2, "repetition", split.repetition)
                        alpha_table.insert(3, "bound_tau", tau)
                        alpha_table.insert(4, "bound_delta", args.bound_delta)
                        alpha_table.insert(5, "bound", bound)
                        alpha_table.insert(6, "alpha_col", alpha_col)
                        alpha_table.insert(7, "calibration", calibration)

                        all_detail.append(detail)
                        all_alpha_tables.append(alpha_table)

    bound_trace_table = pd.concat(all_trace_tables, ignore_index=True) if all_trace_tables else pd.DataFrame()
    detail = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    alpha_summary = summarize_alpha_tables(all_alpha_tables)
    summary_by_split = summarize_detail_by_split(detail)
    summary = summarize_detail_overall(detail)
    summary_by_group = summarize_detail_by_group(detail)

    stableshots_detail = pd.DataFrame()
    stableshots_summary = pd.DataFrame()
    stableshots_by_group = pd.DataFrame()
    stableshots_table1 = pd.DataFrame()
    scaled_table1 = summarize_scaled_table1_comparison(detail, table_thresholds)
    if args.stableshots_trace_metrics:
        stableshots_detail = load_stableshots_trace_metrics(
            path=Path(args.stableshots_trace_metrics),
            wide=wide,
            splits=splits,
            strategy_filter=args.stableshots_strategy,
            scheme_filter=args.stableshots_scheme,
            k_filter=args.stableshots_k,
            policy_filter=args.stableshots_policy,
        )
        stableshots_detail, stableshots_summary, stableshots_by_group, stableshots_table1 = summarize_stableshots_sensitivity(
            stableshots=stableshots_detail,
            taus=taus,
            covered_threshold=args.stableshots_covered_threshold,
            table_thresholds=table_thresholds,
        )

    bound_trace_table.to_csv(output_dir / "bound_trace_table.csv", index=False)
    split_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    alpha_summary.to_csv(output_dir / "grouped_alpha_summary.csv", index=False)
    detail.to_csv(output_dir / "grouped_scaled_bound_trace_metrics.csv", index=False)
    summary_by_split.to_csv(output_dir / "grouped_scaled_bound_summary_by_split.csv", index=False)
    summary.to_csv(output_dir / "grouped_scaled_bound_summary.csv", index=False)
    summary_by_group.to_csv(output_dir / "grouped_scaled_bound_summary_by_group.csv", index=False)
    scaled_table1.to_csv(output_dir / "table1_style_scaled_bound_summary.csv", index=False)
    comparison_frames = [scaled_table1]
    if not stableshots_table1.empty:
        comparison_frames.insert(0, stableshots_table1)
    table1_comparison = pd.concat(comparison_frames, ignore_index=True, sort=False) if comparison_frames else pd.DataFrame()
    table1_comparison.to_csv(output_dir / "table1_style_comparison.csv", index=False)
    if not stableshots_detail.empty:
        stableshots_detail.to_csv(output_dir / "stableshots_sensitivity_trace_metrics.csv", index=False)
        stableshots_summary.to_csv(output_dir / "stableshots_sensitivity_summary.csv", index=False)
        stableshots_by_group.to_csv(output_dir / "stableshots_sensitivity_summary_by_group.csv", index=False)
        stableshots_table1.to_csv(output_dir / "table1_style_stableshots_summary.csv", index=False)

    print("Wrote grouped RQ3 scaling outputs to:")
    print(f"  {output_dir.resolve()}")
    print()
    print("Main summary:")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 220):
        print(summary.sort_values(["bound_tau", "bound", "calibration", "group_mode"]).to_string(index=False))
        print()
        print("Table-I-style comparison summary:")
        if table1_comparison.empty:
            print("  No comparison rows available.")
        else:
            print(table1_comparison.to_string(index=False))
        if not stableshots_summary.empty:
            print()
            print("StableShots detailed all-trace plus covered-only summary:")
            print(stableshots_summary.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze RQ3 bound scaling with global, size, and size-algorithm calibration."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace-metrics", default="", help="Path to trace_metrics.csv from the experiment script.")
    source.add_argument(
        "--fixed-subset-metrics",
        default="",
        help="Path to aggregate_fixed_subset_metrics.csv-style file.",
    )

    parser.add_argument("--output-dir", default="rq3_grouped_scaling")
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--fixed-baselines", default=",".join(map(str, DEFAULT_FIXED_BASELINES)))
    parser.add_argument("--taus", default="0.05", help="Comma-separated TVD targets, e.g. 0.01,0.05,0.10")
    parser.add_argument("--bound-delta", type=float, default=0.05)
    parser.add_argument("--table-thresholds", default="0.01,0.05,0.10", help="TVD thresholds reported in Table-I-style outputs.")
    parser.add_argument("--calibrations", default="median,p75,p90")
    parser.add_argument("--group-modes", default="global,size,size_algorithm")
    parser.add_argument("--split-mode", choices=sorted(ALLOWED_SPLIT_MODES), default="backend_holdout")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-repetitions", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--stableshots-trace-metrics",
        default="",
        help=(
            "Optional path to validation_test_selected_trace_metrics.csv, trace_metrics.csv, "
            "materialized_trace_metrics.csv, aggregate_stableshots_subset_metrics.csv, "
            "or aggregate_policy_subset_metrics.csv. If provided, the script writes plain "
            "Table-I-style StableShots summaries over all held-out rows, plus optional "
            "covered-only sensitivity columns based on --stableshots-covered-threshold."
        ),
    )
    parser.add_argument(
        "--stableshots-strategy",
        default="",
        help=(
            "Optional StableShots strategy or config_id to select when --stableshots-trace-metrics "
            "points to a trace_metrics.csv containing multiple StableShots configs."
        ),
    )
    parser.add_argument(
        "--stableshots-scheme",
        default="uniform",
        help="For aggregate StableShots/policy CSVs, select this weighting scheme before sensitivity analysis.",
    )
    parser.add_argument(
        "--stableshots-k",
        type=int,
        default=1,
        help="For aggregate StableShots/policy CSVs, select this backend-subset size before sensitivity analysis.",
    )
    parser.add_argument(
        "--stableshots-policy",
        default="local_only_offline_aggregate",
        help="For aggregate_policy_subset_metrics.csv, select this policy before sensitivity analysis.",
    )
    parser.add_argument(
        "--stableshots-covered-threshold",
        type=float,
        default=18000.0,
        help="StableShots rows with shots greater than this threshold are excluded from covered-only sensitivity metrics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
