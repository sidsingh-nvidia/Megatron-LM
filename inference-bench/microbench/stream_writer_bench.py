"""Per-token cost of the Megatron inference streaming writer, by position in the output.

For each streamed token the chat endpoint's writer (openai_streaming.py) runs:
  1. detokenizer.update([token])       -- HuggingFaceFastIncrementalDetokenizer
  2. detokenizer.text                  -- "".join over every fragment so far
  3. parser.parse(text)                -- StreamingChatParser over the full text, with the
                                          production parsers (nemotron-v3-reasoning, qwen3-coder-tool)
This streams a ~12k-token reasoning response one token at a time through exactly that path and
reports the mean time of each step per token, bucketed by how far into the response the token is.
It also times the pre-fix `.text` ("".join over every fragment) and checks the current `.text`
is identical to it at every token.

Run inside the serving container from the repo root:
  python inference-bench/microbench/stream_writer_bench.py --tokenizer <hf tokenizer dir>
"""

import argparse
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "inference-bench/rollouts")
sys.path.insert(0, ".")  # repo root (run from it): megatron


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer directory")
    ap.add_argument("--tokens", type=int, default=12000, help="response length in tokens")
    ap.add_argument("--bucket", type=int, default=1000, help="tokens per reported bucket")
    ap.add_argument("--think-in-output", action="store_true",
                    help="start the response with <think> (served chat templates put it in the "
                         "prompt, so by default the response starts inside the reasoning)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from collect_rollouts import load_rows
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
        _streaming_chat_parsers,
    )
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.incremental_detokenizer import (
        HuggingFaceFastIncrementalDetokenizer,
    )

    hf = AutoTokenizer.from_pretrained(args.tokenizer)
    # The detokenizer reads the HF tokenizer through Megatron's wrapper (`_tokenizer.tokenizer`).
    wrapped = SimpleNamespace(_tokenizer=SimpleNamespace(tokenizer=hf), eod=hf.eos_token_id)

    # A reasoning-style response: <think> + math text until --tokens, then </think> + an answer.
    body = "\n\n".join(r["problem"] for r in load_rows("hard_math"))
    reasoning = ""
    while len(hf.encode(reasoning, add_special_tokens=False)) < args.tokens:
        reasoning += body
    opener = "<think>\n" if args.think_in_output else ""
    ids = hf.encode(opener + reasoning, add_special_tokens=False)[: args.tokens - 20]
    ids += hf.encode("\n</think>\n\nThe answer is 42.", add_special_tokens=False)
    prompt = hf.encode("Solve the problem.", add_special_tokens=False)
    print(f"{len(ids)} response tokens", flush=True)

    options = SimpleNamespace(tools=None, tools_requested=False, chat_template_kwargs=None,
                              parallel_tool_calls=True, is_named_tool_choice=False)
    parsers = ["nemotron-v3-reasoning", "qwen3-coder-tool"]
    parser = _streaming_chat_parsers(parsers, options, 1)[0]  # incremental for this request
    full_parser = _streaming_chat_parsers(parsers, options, 1)[0]._full  # the full re-parse
    detok = HuggingFaceFastIncrementalDetokenizer(wrapped, prompt)
    # The pre-fix `.text`: "".join over every delta so far. Kept here to time it and to check that
    # the detokenizer's `.text` (one appended string) is identical to it at every token.
    fragments = []
    mismatches = 0
    delta_mismatches = 0

    buckets = []  # per bucket: [n, detok, text, parse, old_join, text_chars, full_parse]
    clock = time.perf_counter
    for i, tok in enumerate(ids):
        if i % args.bucket == 0:
            buckets.append([0, 0.0, 0.0, 0.0, 0.0, 0, 0.0])
        b = buckets[-1]
        t0 = clock()
        delta = detok.update([tok])
        t1 = clock()
        text = detok.text
        t2 = clock()
        deltas = parser.parse(text, finished=False)
        t3 = clock()
        b[3] += t3 - t2  # the incremental parse alone
        full_deltas = full_parser.parse(text, finished=False)
        b[6] += clock() - t3
        delta_mismatches += deltas != full_deltas
        t3 = clock()
        if delta:
            fragments.append(delta)
        old_text = "".join(fragments)
        t4 = clock()
        mismatches += old_text != text
        b[0] += 1
        b[1] += t1 - t0
        b[2] += t2 - t1
        b[4] += t4 - t3
        b[5] = len(text)

    print(f"{'tokens':>15} {'text KB':>8} {'detok us':>9} {'.text us':>9} {'parse us':>9} "
          f"{'full parse us':>13} {'total us':>9} {'old total us':>12}")
    for k, (n, d, t, p, v, chars, f) in enumerate(buckets):
        if k % max(1, len(buckets) // 12) and k != len(buckets) - 1:
            continue  # print about a dozen rows
        us = lambda x: 1e6 * x / n  # noqa: E731
        total = us(d) + us(t) + us(p)
        print(f"{k * args.bucket:>7}-{k * args.bucket + n:<7} {chars / 1024:>8.1f} {us(d):>9.1f} "
              f"{us(t):>9.1f} {us(p):>9.1f} {us(f):>13.1f} {total:>9.1f} "
              f"{us(d) + us(v) + us(f):>12.1f}")
    print(f"incremental deltas identical to the full parse at every token: {delta_mismatches == 0} "
          f"({delta_mismatches} mismatches)")
    print(f".text identical to the old join at every token: {mismatches == 0} "
          f"({mismatches} mismatches over {len(ids)} tokens)")
    all_n = sum(b[0] for b in buckets)
    mean = 1e6 * sum(b[1] + b[2] + b[3] for b in buckets) / all_n
    old_mean = 1e6 * sum(b[1] + b[4] + b[6] for b in buckets) / all_n
    print(f"mean over the response: {mean:.1f} us/token now ({1e6 / mean:.0f} tokens/s per writer "
          f"thread), {old_mean:.1f} us/token with the old join + full parse "
          f"({1e6 / old_mean:.0f} tokens/s)")


if __name__ == "__main__":
    main()
