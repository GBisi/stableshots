#!/usr/bin/env python3
"""Generate the simplified Q-SE 2027 paper figures from analysis_v7_qse2027."""

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
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def qpu_handles():
    return [
        Line2D([0], [0], color=QPU_COLORS[q], lw=2, label=QPU_LABELS[q])
        for q in QPU_COLORS
    ]


def plot_rq1(metric, ylabel, stem):
    pairs = pd.read_csv(DATA / "pair_medians.csv")
    baseline = pd.read_csv(DATA / "source_baseline.csv").set_index("source")

    fig, ax = plt.subplots(figsize=(6.9, 3.15))
    for (source, destination), g in pairs.groupby(
        ["source", "destination"], sort=True
    ):
        g = g.sort_values("failure")
        base_col = "tvd_median" if metric == "tvd_median" else "shots_median"
        y0 = float(baseline.loc[source, base_col])
        x = [0.0] + g.failure.tolist()
        y = [y0] + g[metric].tolist()

        # Source-colored pre-failure segment, destination-colored continuation.
        ax.plot(
            x[:2], y[:2], color=QPU_COLORS[source], lw=1.1, alpha=0.75
        )
        ax.plot(
            x[1:], y[1:], color=QPU_COLORS[destination], lw=1.1, alpha=0.75
        )
        ax.scatter(
            [0.0], [y0], color=QPU_COLORS[source], s=10, zorder=3
        )
        ax.scatter(
            g.failure, g[metric], color=QPU_COLORS[destination], s=9, zorder=3
        )

    ax.set_xlabel("Nominal failure fraction (NF = no failure)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(
        [0, .1, .25, .5, .75, .9],
        ["NF", ".10", ".25", ".50", ".75", ".90"],
    )
    ax.grid(axis="y", linewidth=.35, alpha=.35)
    ax.legend(
        handles=qpu_handles(), title="QPU color",
        loc="best", ncol=3, frameon=False
    )
    save(fig, stem)


def plot_rq2(metric, ylabel, stem):
    df = pd.read_csv(DATA / "rq2_source_failure.csv")
    fig, ax = plt.subplots(figsize=(6.9, 3.15))

    for source in QPU_COLORS:
        for strategy, style, marker in [
            ("BEST", "-", "o"),
            ("WORST", "--", "x"),
        ]:
            g = df[
                (df.source == source) & (df.strategy == strategy)
            ].sort_values("failure")
            ax.plot(
                g.failure,
                g[metric],
                color=QPU_COLORS[source],
                linestyle=style,
                marker=marker,
                markersize=3.5,
                linewidth=1.15,
                alpha=.9,
            )

    ax.set_xlabel("Nominal failure fraction")
    ax.set_ylabel(ylabel)
    ax.set_xticks(
        [.1, .25, .5, .75, .9],
        [".10", ".25", ".50", ".75", ".90"],
    )
    ax.grid(axis="y", linewidth=.35, alpha=.35)

    strategy_handles = [
        Line2D(
            [0], [0], color="black", lw=1.2, linestyle="-",
            marker="o", markersize=3.5, label="BEST"
        ),
        Line2D(
            [0], [0], color="black", lw=1.2, linestyle="--",
            marker="x", markersize=3.5, label="WORST"
        ),
    ]
    ax.legend(
        handles=qpu_handles() + strategy_handles,
        loc="best", ncol=4, frameon=False
    )
    save(fig, stem)


# Keep the same order throughout the paper: TVD, then physical shots.
plot_rq1("tvd_median", "Median TVD to 200k Aer", "rq1_pair_tvd")
plot_rq1("shots_median", "Median physical shots", "rq1_pair_shots")
plot_rq2("tvd_median", "Median TVD to 200k Aer", "rq2_oracle_tvd")
plot_rq2("shots_median", "Median physical shots", "rq2_oracle_shots")
