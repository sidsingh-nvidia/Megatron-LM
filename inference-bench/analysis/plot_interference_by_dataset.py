"""Prefill interference by dataset, narrow vs wide EP: share of engine time and of steps spent in
interfered decode steps (a decode-only step on a rank whose EP group has a prefill elsewhere), and
the time those steps lose against clean decodes. Means over each run's steady-state windows, from
the step-time histograms (steady.jsonl `step_time_ms`).

    python3 inference-bench/analysis/plot_interference_by_dataset.py --out interference_by_dataset.png
"""

import argparse
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_figures as pf
from plot_narrow_vs_wide_now import layout

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]
DATASETS = [("Math", "#444444", None), ("WildChat", "#CC79A7", "//")]  # as in the OSL CDF figure
# (EP width label, {dataset: run}): Super v3, 32 requests per GPU, 1 h of 5-min windows each.
RUNS = [
    ("Narrow EP", {"Math": "20261104_162838_math_superv3_narrow_ep_nvls_1h_splits8",
                                     "WildChat": "20261104_162835_wildchat_superv3_narrow_ep_nvls_1h_splits8"}),
    ("Wide EP", {"Math": "20261105_104702_math_superv3_wide_ep_nvls_1h_splits8",
                                          "WildChat": "20261105_104702_wildchat_superv3_wide_ep_nvls_1h_splits8"}),
]


def interference(run):
    """Means over windows: (% of time interfered, % of steps interfered, % of time lost)."""
    time_share, step_share, lost = [], [], []
    for line in open(os.path.join(pf.EXPERIMENTS, run, "steady.jsonl")):
        h = json.loads(line)["step_time_ms"]["all"]
        get = lambda k: (h.get(k, {}).get("count", 0), h.get(k, {}).get("mean_ms", 0.0))  # noqa: E731
        (nc, mc), (ni, mi), (np_, mp) = get("decode_clean"), get("decode_interfered"), get("prefill")
        total_t, total_n = nc * mc + ni * mi + np_ * mp, nc + ni + np_
        time_share.append(100 * ni * mi / total_t)
        step_share.append(100 * ni / total_n)
        lost.append(100 * ni * (mi - mc) / total_t)
    return statistics.mean(time_share), statistics.mean(step_share), statistics.mean(lost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Prefill interference in aggregated serving: Math vs WildChat\n"
                    "NVIDIA Nemotron 3 Super, MInf (NVLS), 32 requests in flight per GPU")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vals = {(w, d): interference(runs[d]) for w, runs in RUNS for d, *_ in DATASETS}
    for (w, d), (t, s, l) in vals.items():
        print(f"{w:10s} {d:9s} interfered: {s:5.1f}% of steps, {t:5.1f}% of time, {l:5.1f}% of time lost")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.36
    for ax, metric, ylabel in ((axes[0], 1, "Interfered decode steps (% of steps)"),
                               (axes[1], 0, "Engine time (%)")):
        for i, (dataset, color, hatch) in enumerate(DATASETS):
            xs = [j + (i - 0.5) * width for j in range(len(RUNS))]
            ys = [vals[(w, dataset)][metric] for w, _ in RUNS]
            ax.bar(xs, ys, width * 0.92, color=color, hatch=hatch, edgecolor="white", linewidth=1.5, label=dataset)
            for x, y, (w, _) in zip(xs, ys, RUNS):
                text = f"{y:.1f}%"
                if metric == 0:  # time panel: also the part that is lost against clean decodes
                    text += f"\n({vals[(w, dataset)][2]:.1f}% lost)"
                ax.text(x, y + 1.2, text, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222222")
        # Tick labels: width plus the parallelism read from the runs' config.env.
        ax.set_xticks(range(len(RUNS)), [f"{w}\n({layout(runs['Math'])}, {pf.gpus_of(runs['Math'])} GPUs)"
                                         for w, runs in RUNS], fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#EEEEEE", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Decode steps waiting on another EP rank's prefill", fontsize=11, loc="left",
                      fontweight="bold")
    axes[1].set_title("Engine time in those steps (and time lost)", fontsize=11, loc="left",
                      fontweight="bold")
    axes[0].legend(frameon=False, fontsize=12, loc="upper left")
    fig.suptitle(args.title, fontsize=14, fontweight="bold", x=0.5, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
