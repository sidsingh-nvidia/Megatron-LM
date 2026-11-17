"""Decode-step wall time, MInf vs vLLM, from nsys traces (one GPU each): one stacked bar per engine,
the non-MoE rest of the step followed by the MoE blocks split by what the routed path is doing. Wall time is sliced, not summed, so overlapping streams count once: an
instant goes to the routed-path stage running then (shared equally if several are), or, when the
routed path runs nothing, to "idle" (split by whether the shared expert is running).

    python3 inference-bench/analysis/plot_step_moe_share.py \\
        --minf minf_rank0.sqlite --vllm vllm_node0.sqlite --out step_moe_share.png
"""

import argparse
import collections
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_decode_step_breakdown import SCAN, STEP
from plot_moe_block_overlap import ENGINES, NORM, kernels
from plot_moe_block_timeline import COMM, DENSE, ELEM, GEMM, ROUTE, stage_of

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]

PARTS = [  # label, color, hatch, routed-path stages
    ("Routed expert GEMMs", GEMM, None, {"expert fc1", "expert fc2"}),
    ("EP dispatch / combine (incl. sanitize)", COMM, None,
     {"dispatch", "sanitize", "combine", "dispatch (AG-v)", "combine (RS-v)"}),
    ("Router + latent GEMMs", DENSE, None, {"router", "latent fc1"}),
    ("Top-k, routing, finalize", ROUTE, None, {"top-k", "pack", "routing", "finalize"}),
    ("Routed path idle, shared expert running", "#D55E00", "xx", set()),
    ("Routed path idle, nothing running", "#BBBBBB", "..", set()),
    ("Other kernels (not matched to a stage)", "#DDDDDD", None, set()),
]
IDLE_SHARED, IDLE, OTHER = 4, 5, 6
DISPATCH = 1
METADATA = r"_fused_metadata_kernel"  # MInf's once-per-step dispatcher metadata exchange
REST = ("Rest of the step (Mamba, attention, TP collectives, ...)", "#999999", None)
MOE = ("MoE blocks", GEMM, None)


def part_of(stage):
    return next(i for i, p in enumerate(PARTS) if stage in p[3])


def analyse(engine, ks):
    """(ns per part summed over blocks, block count, steps, step-window ns)."""
    topk, combine, _ = ENGINES[engine]
    acc, nblocks, i = collections.Counter(), 0, 0
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
        events = []
        for s, e, n, g in ks[j + 1:k + 1]:
            s, e = max(s, lo), min(e, hi)
            if e <= s:
                continue
            key = stage_of(engine, n, g)
            if key:
                tag = part_of(key[1]) if key[0] == "routed" else IDLE_SHARED
            else:
                tag = DISPATCH if re.search(METADATA, n) else OTHER
            events += [(s, 1, tag), (e, -1, tag)]
        events.sort()
        active, t = collections.Counter(), lo
        for x, d, tag in events + [(hi, 0, None)]:
            routed = [p for p, c in active.items() if c > 0 and p != IDLE_SHARED]
            if routed:
                for p in routed:
                    acc[p] += (x - t) / len(routed)
            else:
                acc[IDLE_SHARED if active[IDLE_SHARED] > 0 else IDLE] += x - t
            t = x
            if tag is not None:
                active[tag] += d
        nblocks += 1
        i = k + 1
    names = collections.Counter(n for _, _, n, _ in ks)
    steps = names[STEP] if engine == "MInf" else None
    return acc, nblocks, steps, names, ks[-1][1] - ks[0][0]


def step_rows(ks_minf, ks_vllm):
    """{engine: (wall ms/step, [ms/step per PARTS slice])}."""
    res = {}
    for engine, ks in (("MInf", ks_minf), ("vLLM", ks_vllm)):
        res[engine] = analyse(engine, ks)
    scans_per_step = res["MInf"][3][SCAN] / res["MInf"][2]
    rows = {}
    for engine, (acc, nb, steps, names, window) in res.items():
        steps = steps or names[SCAN] / scans_per_step
        parts = [acc[p] / steps / 1e6 for p in range(len(PARTS))]
        wall = window / steps / 1e6
        rows[engine] = (wall, parts)
        print(f"{engine}: {steps:.0f} steps, {nb / steps:.1f} MoE blocks/step, wall {wall:.2f} ms, MoE {sum(parts):.2f} ms, "
              f"rest {wall - sum(parts):.2f} ms; " + ", ".join(f"{PARTS[p][0]} {v:.2f}" for p, v in enumerate(parts)))

    return rows


def draw_step(ax, rows, legend_loc=dict(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)):
    """One stacked bar per engine; returns (vLLM - MInf step ms, of which MoE ms)."""
    # One bar per engine: the non-MoE rest of the step, then the MoE blocks seen from the routed stream.
    # The shared expert is not drawn; where it holds up the routed stream it shows as idle time.
    segments = [
        ("Non-MoE (Mamba, attention, TP collectives, ...)", "#999999", None, lambda w, p: w - sum(p)),
        ("MoE: routed expert GEMMs", GEMM, None, lambda w, p: p[0]),
        ("MoE: EP dispatch / combine (incl. sanitize)", COMM, None, lambda w, p: p[1]),
        ("MoE: router + latent GEMMs", DENSE, None, lambda w, p: p[2]),
        ("MoE: top-k, routing, finalize, other", ROUTE, None, lambda w, p: p[3] + p[OTHER]),
        ("MoE: routed stream idle", "#D55E00", "xx", lambda w, p: p[IDLE_SHARED] + p[IDLE]),
    ]
    order = ["MInf", "vLLM"]
    vals = {e: [f(*rows[e]) for *_, f in segments] for e in order}
    for y, engine in enumerate(order):
        left = 0
        for (label, color, hatch, _), w in zip(segments, vals[engine]):
            ax.barh(y, w, 0.55, left=left, color=color, hatch=hatch, edgecolor="white", linewidth=1.5,
                    label=label if y == 0 else None)
            if w > 0.3:
                ax.text(left + w / 2, y, f"{w:.1f}", ha="center", va="center", fontsize=8.5, color="#222222",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            left += w
        ax.text(left + 0.15, y, f"{left:.2f} ms", va="center", fontsize=9.5, color="#222222", fontweight="bold")
    for y, engine in enumerate(order):
        a, b = vals[engine][0], sum(vals[engine])
        ax.annotate("", (a, y + 0.38), (b, y + 0.38), arrowprops=dict(arrowstyle="|-|", color="#555555", lw=0.8,
                                                                       shrinkA=0, shrinkB=0, mutation_scale=3))
        ax.text((a + b) / 2, y + 0.42, f"MoE blocks {b - a:.2f} ms", ha="center", va="bottom", fontsize=8,
                color="#333333")
    d = [vals["vLLM"][i] - vals["MInf"][i] for i in range(len(segments))]
    dw, dm = sum(d), sum(d[1:])
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(-0.5, 1.75)
    ax.set_xlabel("ms per decode step (wall time)")
    ax.set_xlim(0, max(sum(v) for v in vals.values()) * 1.12)
    ax.grid(axis="x", color="#EEEEEE", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    h, _ = ax.get_legend_handles_labels()
    ax.legend(h, [f"{seg[0]} ({x:+.2f})" for seg, x in zip(segments, d)], frameon=False, fontsize=8, **legend_loc)
    return dw, dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minf", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--title", default="Ultra wide EP (TP8 x DP8, EP64), steady load c=512: one decode step, GPU 0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(12, 3.6))
    dw, dm = draw_step(ax, step_rows(kernels(args.minf), kernels(args.vllm)))
    ax.set_title(f"{args.title}\nvLLM is {dw:.2f} ms slower per step; {dm:.2f} ms ({dm / dw:.0%}) of it is in the MoE blocks",
                 fontsize=10.5, loc="left")
    fig.text(0.01, 0.01, "Legend: vLLM - MInf in ms. Wall time sliced by what the routed stream runs (the shared "
             "expert shows only as routed-stream idle). Under nsys; communication includes waiting on peer ranks.",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
