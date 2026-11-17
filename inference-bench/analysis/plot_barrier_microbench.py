"""Symmetric-memory barrier cost, old vs new, from inference-bench/microbench/barrier_only_bench.py
(GB200, 152 SMs per GPU; barrier kernel minus empty kernel, µs per launch, CUDA graph of 200,
median of replays, max over ranks).

  old: per-block barrier, every block of every rank signals every peer
  new: once per rank, one multicast (multimem.red) arrival per rank per kernel

Data, from the runs of 2026-11-10:
  EP8:  final benchmark (per_block vs end), 2 nodes; 152 blocks measured twice, averaged.
  EP64: earlier sweep (per_block vs last_block_mc, which is the shipped once-per-rank barrier),
        16 nodes; 1 and 152 blocks from the rerun (two measurements each, averaged) because the
        first run's 1-block row is an outlier (16.9 / 23.2 us).

    python3 inference-bench/analysis/plot_barrier_microbench.py --out barrier_microbench.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]
OLD, NEW = "#D55E00", "#0072B2"

BLOCKS = [1, 8, 32, 128, 152]
DATA = {  # world size: (old per-block, new once-per-rank), µs per launch
    "EP8 (2 nodes)": ([3.58, 5.22, 5.63, 7.87, (8.15 + 8.10) / 2], [3.27, 3.27, 3.21, 3.28, (3.09 + 3.30) / 2]),
    "EP64 (16 nodes)": ([(3.85 + 4.06) / 2, 5.90, 6.30, 13.09, (13.63 + 13.66) / 2],
                        [(3.67 + 3.58) / 2, 4.13, 3.59, 3.60, (3.61 + 3.60) / 2]),
}
DECODE_GRID = (0.5, 2.5)  # x positions spanning 4..64 blocks: decode EP kernels launch about one block per token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, (name, (old, new)) in zip(axes, DATA.items()):
        ax.axvspan(*DECODE_GRID, color="#EEEEEE", zorder=0)
        x = range(len(BLOCKS))  # evenly spaced: 128 and 152 would collide on a log axis
        ax.text(sum(DECODE_GRID) / 2, 15.2, "decode EP grids: 4-64 blocks\n(~1 block per token)",
                ha="center", va="top", fontsize=7.5, color="#555555")
        ax.plot(x, old, color=OLD, marker="o", ms=7, lw=2, label="old: per-block barrier")
        ax.plot(x, new, color=NEW, marker="s", ms=7, lw=2, label="new: once per rank (multicast arrival)")
        for i, o, n in zip(x, old, new):
            ax.annotate(f"{o:.1f}", (i, o), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5,
                        color="#333333")
            ax.annotate(f"{n:.1f}", (i, n), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7.5,
                        color="#333333")
        ax.annotate(f"{old[-1] / new[-1]:.1f}x faster\nat 152 blocks", (x[-1], (old[-1] + new[-1]) / 2), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.5, fontweight="bold",
                    color="#222222")
        ax.set_xticks(x, [str(b) for b in BLOCKS])
        ax.set_xlim(-0.4, len(BLOCKS) - 0.6)
        ax.set_xlabel("blocks per kernel (152 = one per SM)")
        ax.set_title(name, fontsize=10, loc="left", fontweight="bold")
        ax.set_ylim(0, 16)
        ax.grid(color="#EEEEEE", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("barrier cost (µs per launch)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, ncol=2, loc="lower center")
    fig.suptitle("Symmetric-memory barrier microbenchmark, GB200: start + end barrier per kernel, 0 errors",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
