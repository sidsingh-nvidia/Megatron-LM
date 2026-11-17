"""Prompt (ISL) and output (OSL) length distributions of the steady-state datasets, as CDFs, from
the finished requests (steady.requests.jsonl) of one run per dataset.

Only finished requests are recorded, and a window ends some long requests before they finish, so
the OSL tail is somewhat under-represented.

    python3 inference-bench/analysis/plot_isl_osl_cdf.py --run Math <run> --run WildChat <run> --out isl_osl.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paper_figures as pf

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]
STYLES = [("#444444", "-"), ("#CC79A7", "--")]  # dark gray / Okabe-Ito reddish purple, off the agg/PD colours; dashes for CVD


def lengths(run):
    isl, osl = [], []
    for line in open(os.path.join(pf.EXPERIMENTS, run, "steady.requests.jsonl")):
        r = json.loads(line)
        isl.append(r["prompt_tokens"])
        osl.append(r["output_tokens"])
    return np.array(isl), np.array(osl)


def draw_cdfs(axes, runs, kinds=("ISL", "OSL")):
    """One CDF panel per kind ("ISL" and/or "OSL", one axis each), one line per (name, run)."""
    xlabels = {"ISL": "Prompt length (ISL, tokens)", "OSL": "Output length (OSL, tokens)"}
    # Legends where the curves leave room: ISL lower right, OSL upper left.
    legend_loc = {"ISL": "lower right", "OSL": "upper left"}
    for (name, run), (color, ls) in zip(runs, STYLES):
        isl, osl = lengths(run)
        for ax, kind in zip(axes, kinds):
            values = isl if kind == "ISL" else osl
            x = np.sort(values)
            y = np.arange(1, len(x) + 1) / len(x)
            p50, p90 = np.percentile(values, [50, 90])
            # The OSL curves carry p50 / p90 markers, so their legend keeps only the mean.
            stats = (f"mean {values.mean():,.0f}" if kind == "OSL"
                     else f"p50 {p50:,.0f}, p90 {p90:,.0f}, mean {values.mean():,.0f}")
            ax.plot(x, y, color=color, ls=ls, lw=2, label=f"{name}: {stats}  (n={len(x):,})")
            if kind == "OSL":  # mark the p50 and p90 output lengths on the curve
                for q, v, mk in ((0.5, p50, "o"), (0.9, p90, "D")):
                    ax.plot(v, q, marker=mk, ms=8, color=color, markeredgecolor="white", markeredgewidth=1.2,
                            zorder=5)
                    ax.annotate(f"p{q * 100:.0f} {v:,.0f}", (v, q), xytext=(8, -3), textcoords="offset points",
                                fontsize=8.5, color="#222222", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            print(f"{name:9s} {kind}: p10 {np.percentile(values, 10):7,.0f}  p50 {p50:7,.0f}  p90 {p90:7,.0f}  "
                  f"p99 {np.percentile(values, 99):7,.0f}  max {values.max():7,.0f}  mean {values.mean():7,.0f}")
    for ax, kind in zip(axes, kinds):
        ax.set_xscale("log")
        ax.set_xlabel(xlabels[kind])
        ax.set_ylabel("Fraction of requests")
        ax.set_ylim(0, 1.02)
        ax.grid(color="#EEEEEE", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc=legend_loc[kind])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", required=True, metavar=("NAME", "RUN"))
    ap.add_argument("--title", default="Steady-state datasets: prompt and output lengths (Ultra, finished requests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    draw_cdfs(axes, args.run)
    fig.suptitle(args.title, fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Finished requests only: the longest outputs are somewhat under-represented.",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
