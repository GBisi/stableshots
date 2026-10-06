#!/usr/bin/env python3
"""Generate the Q-SE 2027 trajectory figures.

Each trajectory starts at the median source state when the failure is injected
and ends at 100% execution progress.  Dotted lines show the no-failure source
endpoint.  RQ1 colors the post-failure segment by destination QPU; RQ2 uses
source-QPU color and line style for the retrospective BEST/WORST quality
bounds.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path("results/qpu_handoff_policy_matrix")
DATA = ROOT / "analysis_v7_qse2027"
FIG = ROOT / "plots_v7_qse2027"
FIG.mkdir(parents=True, exist_ok=True)

QPU_COLORS = {
    "fez": "#0072B2",
    "kyiv": "#D55E00",
    "marrakesh": "#009E73",
    "sherbrooke": "#CC79A7",
    "torino": "#E69F00",
}
QPU_LABELS = {q: q.capitalize() for q in QPU_COLORS}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, stem):
    fig.tight_layout(pad=0.45)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def qpu_handles():
    return [
        Line2D([0], [0], color=QPU_COLORS[q], lw=2, label=QPU_LABELS[q])
        for q in QPU_COLORS
    ]


def setup_axis(ax, ylabel):
    ax.set_xlim(0, 102)
    ax.set_xticks([0, 10, 25, 50, 75, 90, 100])
    ax.set_xlabel("Execution progress (%)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=.35, alpha=.35)


def plot_rq1(metric, ylabel, stem):
    pairs = pd.read_csv(DATA / "pair_medians.csv")
    prefix = pd.read_csv(DATA / "source_failure_prefix.csv")
    base = pd.read_csv(DATA / "source_baseline.csv").set_index("source")

    start_col = (
        "failure_point_tvd_median"
        if metric == "tvd_median"
        else "failure_shots_median"
    )
    base_col = "tvd_median" if metric == "tvd_median" else "shots_median"

    fig, ax = plt.subplots(figsize=(7.15, 2.55))

    # One no-failure reference for each source/failure point.
    for row in prefix.itertuples(index=False):
        source = row.source
        x0 = 100.0 * float(row.actual_failure_fraction_median)
        y0 = float(getattr(row, start_col))
        y_nf = float(base.loc[source, base_col])
        ax.plot(
            [x0, 100.0], [y0, y_nf],
            color=QPU_COLORS[source],
            linestyle=":",
            linewidth=.85,
            alpha=.40,
            zorder=1,
        )

    prefix_idx = prefix.set_index(["source", "failure"])
    for row in pairs.itertuples(index=False):
        source, dest, failure = (
            row.source,
            row.destination,
            float(row.failure),
        )
        p = prefix_idx.loc[(source, failure)]
        x0 = 100.0 * float(p.actual_failure_fraction_median)
        y0 = float(p[start_col])
        y1 = float(getattr(row, metric))

        # Source marker at failure; destination-colored segment afterward.
        ax.plot(
            [x0, 100.0], [y0, y1],
            color=QPU_COLORS[dest],
            linewidth=.95,
            alpha=.58,
            zorder=2,
        )
        ax.scatter(
            [x0], [y0],
            s=9,
            color=QPU_COLORS[source],
            alpha=.75,
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            [100.0], [y1],
            s=8,
            color=QPU_COLORS[dest],
            alpha=.60,
            edgecolors="none",
            zorder=3,
        )

    setup_axis(ax, ylabel)
    styles = [
        Line2D(
            [0], [0], color="black", lw=1.1, linestyle="-",
            label="handoff to destination"
        ),
        Line2D(
            [0], [0], color="black", lw=1.0, linestyle=":",
            label="source if no failure"
        ),
    ]
    ax.legend(
        handles=qpu_handles() + styles,
        title="QPU color",
        loc="best",
        ncol=4,
        frameon=False,
    )
    save(fig, stem)


def plot_rq2(metric, ylabel, stem):
    df = pd.read_csv(DATA / "rq2_source_failure.csv")
    prefix = pd.read_csv(DATA / "source_failure_prefix.csv")
    base = pd.read_csv(DATA / "source_baseline.csv").set_index("source")

    start_col = (
        "failure_point_tvd_median"
        if metric == "tvd_median"
        else "failure_shots_median"
    )
    base_col = "tvd_median" if metric == "tvd_median" else "shots_median"
    prefix_idx = prefix.set_index(["source", "failure"])

    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    for source in QPU_COLORS:
        for failure in sorted(df[df.source == source].failure.unique()):
            p = prefix_idx.loc[(source, float(failure))]
            x0 = 100.0 * float(p.actual_failure_fraction_median)
            y0 = float(p[start_col])
            y_nf = float(base.loc[source, base_col])

            best = df[
                (df.source == source)
                & (df.failure == failure)
                & (df.strategy == "BEST")
            ].iloc[0]
            worst = df[
                (df.source == source)
                & (df.failure == failure)
                & (df.strategy == "WORST")
            ].iloc[0]

            ax.plot(
                [x0, 100.0], [y0, float(best[metric])],
                color=QPU_COLORS[source],
                linestyle="-",
                linewidth=1.05,
                alpha=.78,
                zorder=2,
            )
            ax.plot(
                [x0, 100.0], [y0, float(worst[metric])],
                color=QPU_COLORS[source],
                linestyle="--",
                linewidth=1.0,
                alpha=.68,
                zorder=2,
            )
            ax.plot(
                [x0, 100.0], [y0, y_nf],
                color=QPU_COLORS[source],
                linestyle=":",
                linewidth=.9,
                alpha=.48,
                zorder=1,
            )
            ax.scatter(
                [x0], [y0], color=QPU_COLORS[source], s=11, zorder=3
            )

    setup_axis(ax, ylabel)
    styles = [
        Line2D(
            [0], [0], color="black", lw=1.2, linestyle="-",
            label="BEST final outcome"
        ),
        Line2D(
            [0], [0], color="black", lw=1.2, linestyle="--",
            label="WORST final outcome"
        ),
        Line2D(
            [0], [0], color="black", lw=1.1, linestyle=":",
            label="source if no failure"
        ),
    ]
    ax.legend(
        handles=qpu_handles() + styles,
        loc="best",
        ncol=4,
        frameon=False,
    )
    save(fig, stem)


# Paper order is always TVD first, then physical shots.
plot_rq1("tvd_median", "Median TVD to 200k Aer", "rq1_pair_tvd")
plot_rq1("shots_median", "Median physical shots", "rq1_pair_shots")
plot_rq2("tvd_median", "Median TVD to 200k Aer", "rq2_oracle_tvd")
plot_rq2("shots_median", "Median physical shots", "rq2_oracle_shots")
