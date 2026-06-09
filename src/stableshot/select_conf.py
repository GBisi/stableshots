#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_summary(trace_df: pd.DataFrame, target_tvd: float) -> pd.DataFrame:
    rows = []

    for strategy, group in trace_df.groupby("strategy", sort=False):
        first = group.iloc[0]

        tvd = group["tvd_to_reference"].astype(float)
        shots = group["shots"].astype(float)

        rows.append(
            {
                "strategy": strategy,
                "strategy_family": first.get("strategy_family", ""),
                "config_id": first.get("config_id", ""),
                "batch_size": first.get("batch_size", np.nan),
                "lookback_batches": first.get("lookback_batches", np.nan),
                "stability": first.get("stability", np.nan),
                "epsilon": first.get("epsilon", np.nan),
                "traces": len(group),
                "success_rate": float((tvd <= target_tvd).mean()),
                "median_shots": float(shots.median()),
                "mean_shots": float(shots.mean()),
                "p95_shots": float(shots.quantile(0.95)),
                "max_shots": float(shots.max()),
                "median_tvd": float(tvd.median()),
                "mean_tvd": float(tvd.mean()),
                "p95_tvd": float(tvd.quantile(0.95)),
                "max_tvd": float(tvd.max()),
                "stable_stop_rate": float((group["stop_reason"] == "stable").mean()),
                "max_budget_rate": float((group["stop_reason"] == "max_budget").mean()),
            }
        )

    return pd.DataFrame(rows)


def add_group_robustness(
    summary_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    target_tvd: float,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    out = summary_df.copy()

    for group_col in group_cols:
        worst_rates = []

        for strategy, strategy_df in trace_df.groupby("strategy", sort=False):
            rates = (
                strategy_df.groupby(group_col)["tvd_to_reference"]
                .apply(lambda s: float((s <= target_tvd).mean()))
            )
            worst_rates.append(
                {
                    "strategy": strategy,
                    f"worst_success_rate_by_{group_col}": float(rates.min()),
                }
            )

        rate_df = pd.DataFrame(worst_rates)
        out = out.merge(rate_df, on="strategy", how="left")

    return out


def pareto_frontier(df: pd.DataFrame, cost_col: str, risk_col: str) -> pd.DataFrame:
    candidates = df.sort_values([cost_col, risk_col], ascending=[True, True]).copy()
    frontier_rows = []
    best_risk = float("inf")

    for _, row in candidates.iterrows():
        risk = float(row[risk_col])
        if risk < best_risk:
            frontier_rows.append(row)
            best_risk = risk

    return pd.DataFrame(frontier_rows)


def select_candidates(
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    candidates = summary_df.copy()

    if not args.include_fixed:
        candidates = candidates[candidates["strategy_family"] == "stable_shots"]

    candidates = candidates[candidates["success_rate"] >= args.min_success_rate]

    risk_col = {
        "max": "max_tvd",
        "p95": "p95_tvd",
        "mean": "mean_tvd",
        "median": "median_tvd",
    }[args.risk_metric]

    candidates = candidates[candidates[risk_col] <= args.target_tvd]

    if args.max_median_shots is not None:
        candidates = candidates[candidates["median_shots"] <= args.max_median_shots]

    if args.max_p95_shots is not None:
        candidates = candidates[candidates["p95_shots"] <= args.max_p95_shots]

    for col in args.group_success_cols:
        candidates = candidates[candidates[col] >= args.min_success_rate]

    return candidates


def add_score(df: pd.DataFrame, args: argparse.Namespace, reference_shots: float) -> pd.DataFrame:
    out = df.copy()

    risk_col = {
        "max": "max_tvd",
        "p95": "p95_tvd",
        "mean": "mean_tvd",
        "median": "median_tvd",
    }[args.risk_metric]

    out["score"] = (
        args.weight_cost * (out["median_shots"] / reference_shots)
        + args.weight_fidelity * (out["median_tvd"] / args.target_tvd)
        + args.weight_risk * (out[risk_col] / args.target_tvd)
        + args.weight_cap * out["max_budget_rate"]
    )

    return out


def rank_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.objective == "min_median_shots":
        return df.sort_values(
            ["median_shots", "max_tvd", "median_tvd"],
            ascending=[True, True, True],
        )

    if args.objective == "min_mean_shots":
        return df.sort_values(
            ["mean_shots", "max_tvd", "median_tvd"],
            ascending=[True, True, True],
        )

    if args.objective == "min_risk":
        risk_col = {
            "max": "max_tvd",
            "p95": "p95_tvd",
            "mean": "mean_tvd",
            "median": "median_tvd",
        }[args.risk_metric]

        return df.sort_values(
            [risk_col, "median_shots"],
            ascending=[True, True],
        )

    if args.objective == "balanced":
        return df.sort_values(
            ["score", "median_shots"],
            ascending=[True, True],
        )

    raise ValueError(f"Unsupported objective: {args.objective}")


def print_recommendation(ranked: pd.DataFrame, args: argparse.Namespace) -> None:
    if ranked.empty:
        print("No strategy satisfies the requested constraints.")
        return

    best = ranked.iloc[0]

    print()
    print("Best strategy")
    print("-------------")
    print(f"strategy:          {best['strategy']}")
    print(f"config_id:         {best['config_id']}")
    print(f"target_tvd:        <= {args.target_tvd}")
    print(f"success_rate:      {best['success_rate']:.3f}")
    print(f"median_shots:      {best['median_shots']:.0f}")
    print(f"mean_shots:        {best['mean_shots']:.1f}")
    print(f"p95_shots:         {best['p95_shots']:.0f}")
    print(f"median_tvd:        {best['median_tvd']:.6f}")
    print(f"p95_tvd:           {best['p95_tvd']:.6f}")
    print(f"max_tvd:           {best['max_tvd']:.6f}")
    print(f"stable_stop_rate:  {best['stable_stop_rate']:.3f}")
    print(f"max_budget_rate:   {best['max_budget_rate']:.3f}")

    if "score" in best:
        print(f"score:             {best['score']:.6f}")

    print()
    print(f"Top {args.top_k} strategies")
    print("----------------")
    cols = [
        "strategy",
        "success_rate",
        "median_shots",
        "mean_shots",
        "median_tvd",
        "p95_tvd",
        "max_tvd",
        "stable_stop_rate",
        "max_budget_rate",
    ]

    if "score" in ranked.columns:
        cols.append("score")

    print(ranked[cols].head(args.top_k).to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the best StableShots configuration from experiment results."
    )

    parser.add_argument("--trace-metrics", required=True)
    parser.add_argument("--output-dir", default="strategy_selection")

    parser.add_argument("--target-tvd", type=float, required=True)
    parser.add_argument("--min-success-rate", type=float, default=1.0)

    parser.add_argument(
        "--risk-metric",
        choices=["max", "p95", "mean", "median"],
        default="max",
    )

    parser.add_argument(
        "--objective",
        choices=["min_median_shots", "min_mean_shots", "min_risk", "balanced"],
        default="min_median_shots",
    )

    parser.add_argument("--include-fixed", action="store_true")
    parser.add_argument("--max-median-shots", type=float, default=None)
    parser.add_argument("--max-p95-shots", type=float, default=None)

    parser.add_argument(
        "--group-robustness",
        default="",
        help="Comma-separated group columns, e.g. size,algorithm,backend",
    )

    parser.add_argument("--weight-cost", type=float, default=0.5)
    parser.add_argument("--weight-fidelity", type=float, default=0.2)
    parser.add_argument("--weight-risk", type=float, default=0.3)
    parser.add_argument("--weight-cap", type=float, default=0.0)

    parser.add_argument("--top-k", type=int, default=10)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    trace_path = Path(args.trace_metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_df = pd.read_csv(trace_path)

    required = {
        "trace_id",
        "strategy",
        "strategy_family",
        "shots",
        "tvd_to_reference",
        "stop_reason",
    }

    missing = required - set(trace_df.columns)
    if missing:
        raise ValueError(f"trace_metrics.csv is missing required columns: {sorted(missing)}")

    group_cols = parse_csv(args.group_robustness)
    args.group_success_cols = [f"worst_success_rate_by_{col}" for col in group_cols]

    summary = build_summary(trace_df, args.target_tvd)

    if group_cols:
        summary = add_group_robustness(summary, trace_df, args.target_tvd, group_cols)

    reference_shots = float(trace_df["reference_shots"].max()) if "reference_shots" in trace_df.columns else 20000.0

    candidates = select_candidates(summary, args)
    candidates = add_score(candidates, args, reference_shots)
    ranked = rank_candidates(candidates, args)

    risk_col = {
        "max": "max_tvd",
        "p95": "p95_tvd",
        "mean": "mean_tvd",
        "median": "median_tvd",
    }[args.risk_metric]

    frontier = pareto_frontier(summary, "median_shots", risk_col)

    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    ranked.to_csv(output_dir / "recommended_strategies.csv", index=False)
    frontier.to_csv(output_dir / "pareto_frontier.csv", index=False)

    print_recommendation(ranked, args)
    print()
    print(f"Wrote outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()