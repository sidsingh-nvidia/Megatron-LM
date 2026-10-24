"""Throughput vs interactivity Pareto from steady_client.py runs (steady.jsonl).

One point per concurrency level: x = median tokens/s per user (1000 / median TPOT, TTFT
included), y = output tokens/s per GPU (from the server's /metrics over the window). A thin
whisker reaches left to the p90 TPOT (the slow tail). Up and to the right is better. GPUs per
run come from NODES in the run's config.env (4 GPUs per node) unless given as label=run:gpus.

    python3 inference-bench/analysis/plot_pareto.py "nvls=<run>" "DeepEP=<run>" \
        --title "Super v3, TP4 x EP4" --out inference-bench/analysis/pareto.png
"""

import argparse
import json
import logging
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Okabe-Ito, colorblind-safe, fixed order; distinct markers and line styles too.
STYLES = [("#0072B2", "o", "-"), ("#D55E00", "^", "--"), ("#009E73", "s", ":"),
          ("#CC79A7", "D", "-."), ("#E69F00", "v", (0, (5, 1)))]


def gpus_of(run):
    m = re.search(r"\bNODES=(\d+)", open(os.path.join(run, "config.env")).read())
    return 4 * int(m.group(1)) if m else 4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=run_dir[:gpus]")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, spec in enumerate(args.runs):
        label, rest = spec.split("=", 1)
        run, _, gpus = rest.partition(":")
        gpus = int(gpus) if gpus else gpus_of(run)
        rows = sorted((json.loads(line) for line in open(os.path.join(run, "steady.jsonl"))),
                      key=lambda r: r["concurrency"])
        rows = [r for r in rows if r["tpot_ms_p50"] == r["tpot_ms_p50"]]  # drop NaN (none finished)
        x = [r["user_tok_s_p50"] for r in rows]
        y = [r["output_tok_s"] / gpus for r in rows]
        color, marker, ls = STYLES[i % len(STYLES)]
        ax.plot(x, y, color=color, marker=marker, linestyle=ls, linewidth=2, markersize=8,
                label=f"{label} ({gpus} GPUs)", zorder=3)
        for r, xi, yi in zip(rows, x, y):
            ax.plot([r["user_tok_s_p90_slow"], xi], [yi, yi], color=color, linewidth=1, alpha=0.5)
            ax.annotate(f"c={r['concurrency']}", (xi, yi), textcoords="offset points",
                        xytext=(6, 5 if i % 2 == 0 else -12), fontsize=8, color="#333333")
    ax.set_xlabel("Interactivity: tokens/s per user (median; whisker to p90 TPOT)")
    ax.set_ylabel("Output throughput: tokens/s per GPU")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#dddddd", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    if args.title:
        ax.set_title(args.title, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
