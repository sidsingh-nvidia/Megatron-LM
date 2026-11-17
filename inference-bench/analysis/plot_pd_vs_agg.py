"""PD disagg (prefill/decode disaggregated) vs aggregated serving: the datasets' output
length CDF (from each dataset's aggregated run) on top, then one column per dataset of median TPOT
and of throughput vs p90 interactivity. Same load accounting as plot_narrow_vs_wide_now.py: load per decode (or
aggregated) GPU; throughput per GPU of the whole run, PD prefill pool included.

    python3 inference-bench/analysis/plot_pd_vs_agg.py \\
        --dataset Math <agg run> <PD run> --dataset WildChat <agg run> <PD run> --out pd_vs_agg.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_isl_osl_cdf import draw_cdfs
from plot_narrow_vs_wide_now import MINF, layout, load, total_gpus

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]
PD = "#009E73"  # bluish green, the PD series colour in the other Ultra figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=3, action="append", required=True, metavar=("NAME", "AGG", "PD"))
    ap.add_argument("--title", default="NVIDIA Nemotron 3 Ultra 550B-A55B: wide-EP deployment\n"
                    "aggregated vs disaggregated (PD) serving")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Rows 2-3: one column per dataset (two half-width columns of a 4-column grid each).
    # Row 1: the output-length CDF, centred over the middle two grid columns.
    assert len(args.dataset) == 2, "the layout has two dataset columns"
    # A thin row between the CDF and the TPOT row holds the aggregated / PD disagg legend.
    fig = plt.figure(figsize=(12.4, 12.8))
    grid = fig.add_gridspec(4, 4, height_ratios=[1, 0.07, 1, 1])
    osl_ax = fig.add_subplot(grid[0, 1:3])
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")
    axes = [[None, None]] + [[fig.add_subplot(grid[r, 0:2]), fig.add_subplot(grid[r, 2:4])] for r in (2, 3)]
    for col, (name, agg, pd) in enumerate(args.dataset):
        series = [(f"Aggregated: {layout(agg)} ({total_gpus(agg)} GPUs)", load(agg), MINF, "o"),
                  (f"PD disagg: {layout(pd)} ({total_gpus(pd)} GPUs)", load(pd), PD, "s")]
        tpot, pareto = axes[1][col], axes[2][col]
        for label, pts, color, marker in series:
            per_gpu, tput, inter, t, _ = zip(*pts)
            pareto.plot(inter, tput, color=color, marker=marker, lw=2, ms=7, label=label)
            for x, y, c in zip(inter, tput, per_gpu):
                pareto.annotate(f"{c:g}/GPU", (x, y), xytext=(5, -2), textcoords="offset points", fontsize=7,
                                color="#555555")
            tpot.plot(per_gpu, t, color=color, marker=marker, lw=2, ms=7, label=label)
        # PD disagg vs aggregated TPOT at each shared load.
        agg_pts = {p[0]: p for p in series[0][1]}
        for c, tput, _, t, _ in series[1][1]:
            if c in agg_pts:
                a = agg_pts[c]
                # Above the PD point when PD is slower (agg's line is below it), else below.
                tpot.annotate(f"{100 * (t / a[3] - 1):+.0f}%", (c, t), xytext=(0, 8 if t > a[3] else -14),
                              textcoords="offset points",
                              ha="center", fontsize=8, color=PD, fontweight="bold")
        pareto.set_title(f"{name}: throughput vs p90 interactivity", fontsize=10.5, loc="left", fontweight="bold")
        pareto.set_xlabel("p90 interactivity (tokens/s/user)")
        pareto.set_ylabel("Throughput (tokens/s/GPU, all GPUs)")
        tpot.set_title(f"{name}: median TPOT (labels: PD disagg vs aggregated)", fontsize=10.5, loc="left", fontweight="bold")
        loads = sorted({p[0] for _, pts, *_ in series for p in pts})
        tpot.set_xscale("log", base=2)
        tpot.set_xticks(loads, [f"{c:g}" for c in loads])
        tpot.set_xlabel("Requests in flight per GPU (PD disagg: per decode GPU)")
        tpot.set_ylabel("Median TPOT (ms)")
        for ax in (pareto, tpot):
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
            ax.grid(color="#EEEEEE", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    # TPOT and Pareto rows: one y range per row, so the datasets compare at a glance.
    for row in axes[1:]:
        top = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(0, top)
    # Pareto row: one interactivity (x) range too.
    lo = min(ax.get_xlim()[0] for ax in axes[2])
    hi = max(ax.get_xlim()[1] for ax in axes[2])
    for ax in axes[2]:
        ax.set_xlim(lo, hi)
    # First row: output lengths of the datasets.
    draw_cdfs([osl_ax], [(name, agg) for name, agg, _ in args.dataset], kinds=("OSL",))
    osl_ax.set_title("Dataset output lengths (CDF, finished requests)", fontsize=10.5, loc="left",
                     fontweight="bold")
    # Same two series in every TPOT / Pareto panel: one large legend, above those panels.
    handles, labels = axes[1][0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, frameon=False, fontsize=13, loc="center", ncol=2, markerscale=1.4,
                     handlelength=3)
    fig.suptitle(args.title, fontsize=18, fontweight="bold", x=0.5, ha="center", y=0.995, linespacing=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
