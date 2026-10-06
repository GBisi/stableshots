#!/usr/bin/env python3
"""Generate the Q-SE 2027 complete-execution trajectory figures.

RQ1 paths run from 0% to 100% execution progress, using source-QPU color before
the injected failure and destination-QPU color after the switch. RQ2 uses the
same source prefix but neutral solid/dashed branches for retrospective
BEST/WORST because the oracle-selected destination varies across the
circuit/size cases summarized by each median. RQ2 intentionally omits the
no-failure reference.
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


def source_path(prefix, source, failure, metric):
    """Median source-side path from 0% up to one failure point.

    Source states are observed at the five injected failure checkpoints. For
    shot count, 0% is exactly 0 shots. TVD is undefined before measurements
    exist, so the first observed source TVD is extended to 0% only to show the
    complete source-QPU phase; no additional TVD observation is implied.
    """
    work = prefix[
        (prefix.source == source)
        & (prefix.failure <= failure + 1e-12)
    ].sort_values("failure")
    if work.empty:
        raise RuntimeError(f"no source prefix for {source}/{failure}")

    xs = (100.0 * work.actual_failure_fraction_median.astype(float)).tolist()
    if metric == "tvd_median":
        ys = work.failure_point_tvd_median.astype(float).tolist()
        xs = [0.0] + xs
        ys = [ys[0]] + ys
    else:
        ys = work.failure_shots_median.astype(float).tolist()
        xs = [0.0] + xs
        ys = [0.0] + ys
    return xs, ys


def plot_rq1(metric, ylabel, stem):
    pairs = pd.read_csv(DATA / "pair_medians.csv")
    prefix = pd.read_csv(DATA / "source_failure_prefix.csv")
    base = pd.read_csv(DATA / "source_baseline.csv").set_index("source")

    base_col = "tvd_median" if metric == "tvd_median" else "shots_median"
    fig, ax = plt.subplots(figsize=(7.15, 2.55))

    # One full no-failure source path per QPU.
    for source in QPU_COLORS:
        xs, ys = source_path(prefix, source, 0.9, metric)
        xs = xs + [100.0]
        ys = ys + [float(base.loc[source, base_col])]
        ax.plot(
            xs, ys,
            color=QPU_COLORS[source],
            linestyle=":",
            linewidth=1.0,
            alpha=.50,
            zorder=1,
        )

    prefix_idx = prefix.set_index(["source", "failure"])
    for row in pairs.itertuples(index=False):
        source = row.source
        dest = row.destination
        failure = float(row.failure)
        p = prefix_idx.loc[(source, failure)]
        xf = 100.0 * float(p.actual_failure_fraction_median)
        yf = float(
            p.failure_point_tvd_median
            if metric == "tvd_median"
            else p.failure_shots_median
        )
        y1 = float(getattr(row, metric))

        xs, ys = source_path(prefix, source, failure, metric)
        xs[-1], ys[-1] = xf, yf
        ax.plot(
            xs, ys,
            color=QPU_COLORS[source],
            linewidth=.90,
            alpha=.42,
            zorder=2,
        )
        ax.plot(
            [xf, 100.0], [yf, y1],
            color=QPU_COLORS[dest],
            linewidth=1.05,
            alpha=.62,
            zorder=3,
        )
        ax.scatter(
            [xf], [yf],
            s=13,
            facecolor=QPU_COLORS[source],
            edgecolor="black",
            linewidth=.25,
            alpha=.90,
            zorder=4,
        )
        ax.scatter(
            [100.0], [y1],
            s=9,
            color=QPU_COLORS[dest],
            alpha=.65,
            edgecolors="none",
            zorder=4,
        )

    setup_axis(ax, ylabel)
    styles = [
        Line2D([0], [0], color="black", lw=1.2, linestyle="-",
               label="execution path"),
        Line2D([0], [0], color="black", lw=1.0, linestyle=":",
               label="source without failure"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=4,
               label="failure / QPU switch"),
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
    prefix_idx = prefix.set_index(["source", "failure"])

    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    for source in QPU_COLORS:
        for failure in sorted(df[df.source == source].failure.unique()):
            p = prefix_idx.loc[(source, float(failure))]
            xf = 100.0 * float(p.actual_failure_fraction_median)
            yf = float(
                p.failure_point_tvd_median
                if metric == "tvd_median"
                else p.failure_shots_median
            )
            xs, ys = source_path(prefix, source, float(failure), metric)
            xs[-1], ys[-1] = xf, yf

            ax.plot(
                xs, ys,
                color=QPU_COLORS[source],
                linewidth=.95,
                alpha=.58,
                zorder=2,
            )
            ax.scatter(
                [xf], [yf],
                s=13,
                facecolor=QPU_COLORS[source],
                edgecolor="black",
                linewidth=.25,
                alpha=.90,
                zorder=4,
            )

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

            # Neutral post-failure styles: each median can summarize different
            # selected destinations across circuit/size configurations.
            ax.plot(
                [xf, 100.0], [yf, float(best[metric])],
                color="black",
                linestyle="-",
                linewidth=1.05,
                alpha=.62,
                zorder=3,
            )
            ax.plot(
                [xf, 100.0], [yf, float(worst[metric])],
                color="0.30",
                linestyle="--",
                linewidth=1.00,
                alpha=.62,
                zorder=3,
            )

    setup_axis(ax, ylabel)
    styles = [
        Line2D([0], [0], color="black", lw=1.2, linestyle="-",
               label="BEST final outcome"),
        Line2D([0], [0], color="0.30", lw=1.2, linestyle="--",
               label="WORST final outcome"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=4,
               label="failure / QPU switch"),
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
