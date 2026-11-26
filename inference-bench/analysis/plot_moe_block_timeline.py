"""Timeline of one MoE block, MInf vs vLLM, from nsys traces (one GPU each): the routed path and the
shared expert as two lanes, each split into logical stages. A stage's start and end are medians over
every MoE block in the trace, as offsets from the block start (end of the pre-router norm).

    python3 inference-bench/analysis/plot_moe_block_timeline.py \\
        --minf minf_rank0.sqlite --vllm vllm_node0.sqlite --out moe_timeline.png
"""

import argparse
import collections
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_moe_block_overlap import ENGINES, NORM, kernels

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]

GEMM, DENSE, COMM, ROUTE, ELEM = "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7"
KINDS = [(GEMM, "Routed expert GEMM"), (DENSE, "Dense GEMM"), (COMM, "Communication"),
         (ROUTE, "Routing bookkeeping"), (ELEM, "Elementwise")]
# Per engine: (stage, color, name regex, gridX or None); a stage may list several kernels, e.g. a
# split-K GEMM and its reduce (told apart by grid).
STAGES = {
    "vLLM": {
        "routed": [("router", DENSE, r"tss_64x8_64x16", 64), ("router", DENSE, r"splitKreduce.*float, float, float, float", 16),
                   ("latent fc1", DENSE, r"64x8_64x16_2x2_h_bz_splitK", 128), ("latent fc1", DENSE, r"splitKreduce", 64),
                   ("top-k", ROUTE, r"topk", None), ("dispatch", COMM, r"moeA2APrepareDispatch|moeA2ADispatch", None),
                   ("sanitize", ROUTE, r"Sanitize", None), ("routing", ROUTE, r"routingIndices", None),
                   ("expert fc1", GEMM, r"bmm_", 40), ("expert fc2", GEMM, r"bmm_", 16), ("finalize", ELEM, r"finalize", None),
                   ("combine", COMM, r"moeA2APrepareCombine|moeA2ACombine", None)],
        "shared": [("up", DENSE, r"2x1_2cta_v_bz_splitK", 1280), ("up", DENSE, r"splitKreduce", 320),
                   ("act", ELEM, r"pow_relu", None),
                   ("down", DENSE, r"2x1_2cta_v_bz_splitK", 896), ("down", DENSE, r"splitKreduce", 256)],
    },
    "MInf": {
        "routed": [("router", DENSE, r"tss_64x8_64x16", 40), ("router", DENSE, r"splitKreduce.*float, float, float, float", 16),
                   ("top-k", ROUTE, r"topk", None),
                   ("latent fc1", DENSE, r"64x8_64x16_4x1_v_bz_splitK", 96), ("latent fc1", DENSE, r"splitKreduce", 64), ("pack", ROUTE, r"pack_routing", None),
                   ("dispatch (AG-v)", COMM, r"all_gatherv", None), ("routing", ROUTE, r"routingIndices", None),
                   ("expert fc1", GEMM, r"bmm_", 40), ("expert fc2", GEMM, r"bmm_", 16), ("finalize", ELEM, r"_finalize", None),
                   ("combine (RS-v)", COMM, r"reduce_scatter_v", None)],
        "shared": [("TP AG", COMM, r"_multimem_all_gather_kernel", None), ("up", DENSE, r"2x2_2cta_h_bz_splitK", 120), ("up", DENSE, r"splitKreduce", 40),
                   ("act", ELEM, r"pow_relu", None), ("down", DENSE, r"128x32_64x11", 128),
                   ("TP RS", COMM, r"_multimem_reduce_scatter_kernel", None)],
    },
}
LANES = ["routed", "shared"]
TIMED = {"dispatch", "sanitize", "combine", "dispatch (AG-v)", "combine (RS-v)", "up", "down", "TP AG", "TP RS"}


def stage_of(engine, name, grid):
    for lane in LANES:
        for stage, _, pattern, g in STAGES[engine][lane]:
            if re.search(pattern, name) and (g is None or grid == g):
                return lane, stage
    return None


def timeline(engine, ks):
    """{(lane, stage): (median start, median end)} in µs from the block start, and median block length."""
    topk, combine, _ = ENGINES[engine]
    spans, lengths, i = collections.defaultdict(lambda: ([], [])), [], 0
    while i < len(ks):
        if not re.search(topk, ks[i][2]):
            i += 1
            continue
        j = i
        while j > 0 and not re.search(NORM, ks[j][2]):
            j -= 1
        k = i
        while k < len(ks) and not re.search(combine, ks[k][2]):
            k += 1
        if j == 0 or k == len(ks):
            break
        lo, hi = ks[j][1], ks[k][1]
        block = {}
        for s, e, n, g in ks[j + 1:k + 1]:
            key = stage_of(engine, n, g)
            if key is None:
                continue
            a, b = block.get(key, (s, e))
            block[key] = (min(a, s), max(b, e))
        for key, (a, b) in block.items():
            spans[key][0].append((a - lo) / 1e3)
            spans[key][1].append((b - lo) / 1e3)
        lengths.append((hi - lo) / 1e3)
        i = k + 1
    return {key: (statistics.median(a), statistics.median(b)) for key, (a, b) in spans.items()}, statistics.median(lengths), len(lengths)


def marker(ax, x, y, k):
    ax.text(x, y, str(k), ha="center", va="center", fontsize=7, color="white", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.18", fc="#222222", ec="none"))


def annotate(ax, engine, spans):
    """Numbered call-outs for the three places the blocks differ, plus a key at the right."""
    span = lambda lane, *stages: sum(spans[(lane, st)][1] - spans[(lane, st)][0] for st in stages)
    if engine == "vLLM":
        comm, shared = ("dispatch", "sanitize", "combine"), ("up", "act", "down")
    else:
        comm, shared = ("dispatch (AG-v)", "combine (RS-v)"), ("up", "act", "down")
    # 1: the routed path waiting between top-k / pack and dispatch.
    before = "top-k" if engine == "vLLM" else "pack"
    idle0, idle1 = spans[("routed", before)][1], spans[("routed", comm[0])][0]
    routed = sorted(v for (lane, _), v in spans.items() if lane == "routed")
    end = routed[0][1]
    for a, b in routed[1:]:  # every gap on the routed lane: nothing on it is running
        if a - end > 2:
            ax.barh(1, a - end, 0.6, left=end, color="none", edgecolor="#D55E00", hatch="..", linewidth=1)
            ax.text((a + end) / 2, 1, f"idle {a - end:.0f} µs" if a - end > 12 else f"{a - end:.0f}", ha="center",
                    va="center", fontsize=7.5, color="#D55E00", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        end = max(end, b)
    marker(ax, idle0 + 1.5, 0.62, 1)
    for st in comm:
        marker(ax, spans[("routed", st)][0] + 1.5, 0.62, 2)
    marker(ax, min(a for (lane, _), (a, _) in spans.items() if lane == "shared") - 3, 0, 3)
    gemm = span("shared", "up", "down")
    if engine == "vLLM":
        lines = [f"routed path idle {idle1 - idle0:.0f} µs while the shared-expert GEMM runs",
                 f"dispatch + sanitize + combine: {span('routed', *comm):.0f} µs",
                 f"shared expert NOT split: full weights on every GPU,\n8 tokens each: {gemm:.0f} µs of GEMM"]
    else:
        lines = ["no idle wait: dispatch starts right after pack",
                 f"AG-v + RS-v: {span('routed', *comm):.0f} µs",
                 f"shared expert split 8 ways (TP): {gemm:.0f} µs of GEMM\n+ {span('shared', 'TP AG', 'TP RS'):.0f} µs TP all-gather / reduce-scatter"]
    for k, line in enumerate(lines):
        y = 1.55 - k * 0.8
        marker(ax, 206, y, k + 1)
        ax.text(210, y, line, va="center", fontsize=8, color="#222222")


def draw_timeline(axes, data):
    """One axis per engine, in the order of `data`: the routed and shared lanes with call-outs."""
    for ax, engine in zip(axes, data):
        spans, length, n = data[engine]
        print(f"{engine}: {n} blocks, median block {length:.1f} us")
        for y, lane in enumerate(LANES):
            for stage, color in dict((st, c) for st, c, _, _ in STAGES[engine][lane]).items():
                if (lane, stage) not in spans:
                    continue
                a, b = spans[(lane, stage)]
                print(f"  {lane:6s} {stage:16s} {a:6.1f} -> {b:6.1f}  ({b - a:5.1f})")
                ax.barh(1 - y, b - a, 0.6, left=a, color=color, edgecolor="white", linewidth=1.5)
                label = f"{stage} {b - a:.0f} µs" if stage in TIMED else stage
                wide = b - a >= 1.15 * len(label) + 2  # µs the label needs inside the bar
                below = lane == "shared"  # small shared-lane labels hang below, clear of the routed lane
                ax.text((a + b) / 2, 1 - y + (0 if wide else -0.36 if below else 0.36), label,
                        ha="center" if wide else "left", va="center" if wide else "top" if below else "bottom",
                        fontsize=7.5 if wide else 6.5, color="white" if wide and color in (GEMM, ROUTE) else "#222222",
                        rotation=0 if wide else -35 if below else 35, rotation_mode="default" if wide else "anchor")
        annotate(ax, engine, spans)
        ax.axvline(length, color="#555555", lw=1, ls="--")
        ax.text(length + 1.5, 0.5, f"block ends\n{length:.0f} µs", va="center", fontsize=8, color="#333333")
        ax.set_yticks([1, 0], ["routed path", "shared expert"])
        ax.set_ylim(-1.0, 2.1)
        ax.set_title(engine, fontsize=10, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#EEEEEE", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("µs from MoE block start (end of the pre-router norm)")
    axes[-1].set_xlim(0, 300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minf", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--title", default="Ultra wide EP (TP8 x DP8, EP64), steady load c=512: one MoE block, GPU 0 (medians)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = {engine: timeline(engine, kernels(path)) for engine, path in (("vLLM", args.vllm), ("MInf", args.minf))}
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.9), sharex=True)
    draw_timeline(axes, data)
    saved = data["vLLM"][1] - data["MInf"][1]
    args.title += f"\nMInf's MoE block is {saved:.0f} µs shorter; x48 MoE layers = {saved * 48 / 1e3:.1f} ms per decode step"
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in KINDS]
    fig.legend(handles, [l for _, l in KINDS], frameon=False, fontsize=8, ncol=5, loc="lower center")
    fig.suptitle(args.title, fontsize=10.5, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Under nsys; communication kernels include waiting on peer ranks.", fontsize=7,
             color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
