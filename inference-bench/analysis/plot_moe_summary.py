"""One figure, MInf vs vLLM, from nsys traces (one GPU each): the decode-step breakdown on top (one
stacked bar per engine) and, below it, the timeline of one MoE block for each engine (routed and
shared-expert lanes, medians over all blocks).

    python3 inference-bench/analysis/plot_moe_summary.py \\
        --minf minf_rank0.sqlite --vllm vllm_node0.sqlite --out moe_summary.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_moe_block_overlap import kernels
from plot_moe_block_timeline import KINDS, draw_timeline, timeline
from plot_step_moe_share import draw_step, step_rows

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minf", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--title", default="Ultra wide EP (TP8 x DP8, EP64), steady load c=512, GPU 0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ks = {"MInf": kernels(args.minf), "vLLM": kernels(args.vllm)}
    fig = plt.figure(figsize=(14, 11))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.75], hspace=0.62)
    step_ax = fig.add_subplot(outer[0])
    dw, dm = draw_step(step_ax, step_rows(ks["MInf"], ks["vLLM"]),
                       legend_loc=dict(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3))
    step_ax.set_title(f"(a) One decode step: vLLM is {dw:.2f} ms slower; {dm:.2f} ms ({dm / dw:.0%}) of it is in "
                      "the MoE blocks", fontsize=10.5, loc="left")

    data = {engine: timeline(engine, ks[engine]) for engine in ("vLLM", "MInf")}
    inner = outer[1].subgridspec(2, 1, hspace=0.3)
    axes = [fig.add_subplot(inner[0])]
    axes.append(fig.add_subplot(inner[1], sharex=axes[0]))
    draw_timeline(axes, data)
    saved = data["vLLM"][1] - data["MInf"][1]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in KINDS]
    fig.legend(handles, [l for _, l in KINDS], frameon=False, fontsize=8, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.015))
    fig.suptitle(args.title, fontsize=11.5, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.003, "(a) legend: vLLM - MInf in ms; wall time sliced by what the routed stream runs, so the "
             "shared expert shows only as routed-stream idle. Under nsys; communication kernels include waiting on "
             "peer ranks.", fontsize=7, color="#555555")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08)
    fig.text(axes[0].get_position().x0, axes[0].get_position().y1 + 0.03,
             f"(b) One MoE block (medians over all blocks): MInf's is {saved:.0f} µs shorter", fontsize=10.5, va="bottom")
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
