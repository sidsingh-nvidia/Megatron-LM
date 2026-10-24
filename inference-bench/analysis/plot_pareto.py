"""Throughput vs interactivity Pareto from steady_client.py runs (steady.jsonl), in the PSSG style.

One point per concurrency level: x = median tokens/s per user (1000 / median TPOT, TTFT
included), y = output tokens/s per GPU (from the server's /metrics over the window). A thin
whisker reaches left to the p90 TPOT (the slow tail). Up and to the right is better. GPUs per
run come from NODES in the run's config.env (4 GPUs per node) unless given as label=run:gpus.

Needs pssgplot (https://github.com/hpcgroup/pssg-plots); install a clone in editable mode so it
finds its bundled Gill Sans font: `pip install -e pssg-plots/`.

    python3 inference-bench/analysis/plot_pareto.py "NVLS=<run>" "DeepEP=<run>" \
        --title "Super v3, narrow EP" --out inference-bench/analysis/pareto.pdf
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from pssgplot import PALETTE, LinePlot, PlotEnvironment  # noqa: E402


def gpus_of(run):
    m = re.search(r"\bNODES=(\d+)", open(os.path.join(run, "config.env")).read())
    return 4 * int(m.group(1)) if m else 4


def load(spec):
    label, rest = spec.split("=", 1)
    run, _, gpus = rest.partition(":")
    gpus = int(gpus) if gpus else gpus_of(run)
    rows = []
    for line in open(os.path.join(run, "steady.jsonl")):
        r = json.loads(line)
        if r["tpot_ms_p50"] != r["tpot_ms_p50"]:  # NaN: no request finished in the window
            continue
        rows.append({"config": label, "c": r["concurrency"], "user_tok_s": r["user_tok_s_p50"],
                     "user_tok_s_p90": r["user_tok_s_p90_slow"],
                     "tok_s_gpu": r["output_tok_s"] / gpus})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=run_dir[:gpus]")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figsize", type=float, nargs=2, default=(5, 3.5))
    args = ap.parse_args()

    df = pd.DataFrame([row for spec in args.runs for row in load(spec)])
    labels = list(dict.fromkeys(df["config"]))
    with PlotEnvironment():
        plot = LinePlot()
        ax = plot.plot(
            data=df, x="user_tok_s", y="tok_s_gpu", hue="config", style="config",
            hue_order=labels, style_order=labels, sort=False,
            xlabel="Interactivity (tokens/s/user)", ylabel="Throughput (tokens/s/GPU)",
            title=args.title, figsize=tuple(args.figsize),
            # pssgplot trims the margins to zero; leave headroom so edge points are not clipped.
            xlim=(0, 1.1 * df["user_tok_s"].max()), ylim=(0, 1.12 * df["tok_s_gpu"].max()),
        )
        # legend=True passes ncol=None to ax.legend, which newer matplotlib rejects; move the
        # seaborn legend instead (upper right stays clear of the high-throughput end).
        sns.move_legend(ax, "upper right", title=None, frameon=False)
        # p90 whiskers and concurrency labels, in each series' color.
        for i, label in enumerate(labels):
            sub = df[df["config"] == label].sort_values("c")
            color = PALETTE[i % len(PALETTE)]
            for _, r in sub.iterrows():
                ax.plot([r["user_tok_s_p90"], r["user_tok_s"]], [r["tok_s_gpu"]] * 2,
                        color=color, linewidth=1, alpha=0.6)
                ax.annotate(f"{r['c']}", (r["user_tok_s"], r["tok_s_gpu"]), textcoords="offset points",
                            xytext=(5, 4 if i % 2 == 0 else -10), fontsize=7, color=color)
        plot.fig.tight_layout()
        plot.save(args.out, dpi=200)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
