"""Throughput vs interactivity Pareto from steady_client.py runs (steady.jsonl), in the PSSG style.

One point per concurrency level: x = median tokens/s per user (1000 / median TPOT, TTFT
included), y = output tokens/s per GPU (from the server's /metrics over the window). A thin
whisker reaches left to the p90 TPOT (the slow tail). Up and to the right is better. GPUs per
run come from NODES in the run's config.env (4 GPUs per node) unless given as label=run:gpus.

Needs pssgplot (https://github.com/hpcgroup/pssg-plots); install a clone in editable mode so it
finds its bundled Gill Sans font: `pip install -e pssg-plots/`.

    python3 inference-bench/analysis/plot_pareto.py "NVLS/wide=<run>" "DeepEP/wide=<run>" \
        "NVLS/narrow=<run>" "DeepEP/narrow=<run>" --title "Super v3" --out pareto.pdf

Labels "<dispatcher>/<wide|narrow>" get fixed encodings: NVLS blue, DeepEP orange, NVL-A2A green;
wide EP solid, narrow EP dashed. Other labels get the PSSG palette in order.
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pssgplot import PALETTE, PlotEnvironment  # noqa: E402


# Fixed encodings for "dispatcher/ep=run" labels: color and marker by dispatcher, dashes by EP
# width. Other labels get the PSSG palette in order and solid lines.
DISPATCHER_STYLE = {"NVLS": ("#0072B2", "o"), "DeepEP": ("#E69F00", "^"), "NVL-A2A": ("#009E73", "s")}
EP_DASHES = {"wide": "-", "narrow": "--"}


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


def style_of(label, i):
    """(color, marker, linestyle, legend text) for a series label."""
    dispatcher, _, ep = label.partition("/")
    if dispatcher in DISPATCHER_STYLE and ep in EP_DASHES:
        color, marker = DISPATCHER_STYLE[dispatcher]
        return color, marker, EP_DASHES[ep], f"{dispatcher}, {ep} EP"
    return PALETTE[i % len(PALETTE)], "o", "-", label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=run_dir[:gpus]; label 'NVLS/wide', 'DeepEP/narrow', ...")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figsize", type=float, nargs=2, default=(5, 3.5))
    args = ap.parse_args()

    df = pd.DataFrame([row for spec in args.runs for row in load(spec)])
    labels = list(dict.fromkeys(df["config"]))
    with PlotEnvironment():
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        for i, label in enumerate(labels):
            sub = df[df["config"] == label].sort_values("c")
            color, marker, ls, text = style_of(label, i)
            ax.plot(sub["user_tok_s"], sub["tok_s_gpu"], color=color, marker=marker, linestyle=ls,
                    markeredgewidth=0, label=text, zorder=3)
            for _, r in sub.iterrows():
                # Whisker to the p90 TPOT (the slow tail), and the concurrency.
                ax.plot([r["user_tok_s_p90"], r["user_tok_s"]], [r["tok_s_gpu"]] * 2,
                        color=color, linewidth=1, alpha=0.6)
                ax.annotate(f"{r['c']}", (r["user_tok_s"], r["tok_s_gpu"]), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, color=color)
        # PSSG axis style (as pssgplot's LinePlot), with headroom so edge points are not clipped.
        ax.set_xlim(0, 1.1 * df["user_tok_s"].max())
        ax.set_ylim(0, 1.12 * df["tok_s_gpu"].max())
        ax.set_xlabel("Interactivity (tokens/s/user)")
        ax.set_ylabel("Throughput (tokens/s/GPU)")
        if args.title:
            ax.set_title(args.title)
        ax.yaxis.grid(linestyle="dotted", zorder=0)
        ax.spines["left"].set_color("#606060")
        ax.spines["bottom"].set_color("#606060")
        ax.tick_params(axis="both", direction="in", color="#606060")
        ax.legend(loc="upper right", frameon=False, handlelength=3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=200)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
