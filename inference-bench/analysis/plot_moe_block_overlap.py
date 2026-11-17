"""Where the wall time of a MoE block goes, MInf vs vLLM, from nsys traces (one GPU each), with
overlapping streams handled by slicing time instead of summing kernels.

A MoE block runs from the end of the norm before the router to the end of the combine. Within it,
each instant is labelled by what is running: only routed-path kernels (router, latent projection,
top-k, dispatch, expert GEMMs, combine), only shared-expert kernels, both, or nothing. The four
slices add up to the block's wall time. Averaged over every MoE block in the trace.

    python3 inference-bench/analysis/plot_moe_block_overlap.py \\
        --minf minf_rank0.sqlite --vllm vllm_node0.sqlite --out moe_block.png
"""

import argparse
import re
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]

# (top-k kernel, combine kernel, shared-expert kernels as (name regex, gridX or None)).
ENGINES = {
    "vLLM": (
        r"single_group_topk",
        r"moeA2ACombineKernel",
        [(r"nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK", 1280),  # up projection
         (r"nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK", 896),  # down projection
         (r"splitKreduce", 320), (r"splitKreduce", 256), (r"pow_relu", None)],
    ),
    "MInf": (
        r"_sigmoid_topk_kernel",
        r"_multimem_reduce_scatter_v_kernel",
        [(r"_multimem_all_gather_kernel", None), (r"nvjet_sm100_tst_64x32_64x16_2x2_2cta_h_bz_splitK", 120),
         (r"splitKreduce", 40), (r"pow_relu", None), (r"nvjet_sm100_tst_128x32_64x11", 128),
         (r"_multimem_reduce_scatter_kernel", None)],
    ),
}
NORM = r"[Nn]orm"
SLICES = [  # label, color, hatch
    ("Routed path only", "#0072B2", None),
    ("Routed path + shared expert (overlapped)", "#56B4E9", "//"),
    ("Shared expert only (routed path idle)", "#D55E00", "xx"),
    ("Nothing running", "#BBBBBB", None),
]


def kernels(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    dev = min(d for (d,) in db.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    return [(s, e, names[n], g) for s, e, n, g in db.execute(
        "SELECT start, end, demangledName, gridX FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = ? ORDER BY start",
        (dev,))]


def blocks(ks, topk, combine, shared):
    """Per MoE block: ns in each slice, and summed shared-expert kernel time."""
    is_shared = lambda n, g: any(re.search(p, n) and (grid is None or g == grid) for p, grid in shared)
    out, i = [], 0
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
            if e > s:
                kind = 1 if is_shared(n, g) else 0
                events += [(s, 1, kind), (e, -1, kind)]
        events.sort()
        active, t, acc = [0, 0], lo, [0, 0, 0, 0]
        for x, d, kind in events + [(hi, 0, 0)]:
            r, sh = active[0] > 0, active[1] > 0
            acc[0 if r and not sh else 1 if r else 2 if sh else 3] += x - t
            t = x
            active[kind] += d
        se = sum(min(e, hi) - max(s, lo) for s, e, n, g in ks[j + 1:k + 1] if is_shared(n, g))
        out.append((acc, se))
        i = k + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minf", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--layers", type=int, default=48, help="MoE layers per decode step, to scale to ms/step")
    ap.add_argument("--title", default="Ultra wide EP (TP8 x DP8, EP64), steady load c=512: one MoE block, GPU 0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = {}
    for engine, path in (("vLLM", args.vllm), ("MInf", args.minf)):
        b = blocks(kernels(path), *ENGINES[engine])
        n = len(b)
        rows[engine] = ([sum(a[0][s] for a in b) / n / 1e3 for s in range(4)], sum(a[1] for a in b) / n / 1e3)
        sl, se = rows[engine]
        print(f"{engine}: {n} blocks, wall {sum(sl):.1f} us = " + ", ".join(f"{SLICES[s][0]} {v:.1f}" for s, v in enumerate(sl))
              + f"; shared-expert kernel time {se:.1f} us")

    fig, ax = plt.subplots(figsize=(10, 3.0))
    order = ["MInf", "vLLM"]
    for y, engine in enumerate(order):
        left = 0
        for s, (label, color, hatch) in enumerate(SLICES):
            w = rows[engine][0][s]
            ax.barh(y, w, 0.55, left=left, color=color, hatch=hatch, edgecolor="white", linewidth=1.5,
                    label=label if y == 0 else None)
            if w > 6:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=9, color="#222222",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            left += w
        ax.text(left + 2, y, f"{left:.0f} µs  (x{args.layers} = {left * args.layers / 1e3:.2f} ms/step)",
                va="center", fontsize=9, color="#333333")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("µs per MoE block (wall time, sliced by what is running)")
    ax.set_title(args.title, fontsize=10, loc="left")
    ax.set_xlim(0, max(sum(r[0]) for r in rows.values()) * 1.32)
    ax.grid(axis="x", color="#EEEEEE", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.text(0.01, 0.01, "Block = end of the pre-router norm to end of the combine. Under nsys; collective kernels "
             "include waiting on peer ranks.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
