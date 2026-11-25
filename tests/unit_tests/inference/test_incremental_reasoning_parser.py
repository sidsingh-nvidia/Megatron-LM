# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""IncrementalReasoningChatParser streams exactly the deltas of StreamingChatParser."""

import random
from types import SimpleNamespace

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _streaming_chat_parsers,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.openai_streaming import (
    IncrementalReasoningChatParser,
    StreamingChatParser,
)

_PARSERS = ["nemotron-v3-reasoning", "qwen3-coder-tool"]
# Pieces that build every structure the parsers react to, including markers split across deltas.
_PIECES = ["<think>", "</think>", "<", "/", "think", ">", "<thi", "nk>", "</thi", "a", "bc", " ", "\n"]


def _options(**overrides):
    fields = dict(
        tools=None,
        tools_requested=False,
        chat_template_kwargs=None,
        parallel_tool_calls=True,
        is_named_tool_choice=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _pair(parsers=_PARSERS, **overrides):
    """A fresh incremental parser and a fresh full parser for the same request."""
    incremental = _streaming_chat_parsers(parsers, _options(**overrides), 1)[0]
    full = _streaming_chat_parsers(parsers, _options(**overrides), 1)[0]
    assert isinstance(incremental, IncrementalReasoningChatParser)
    return incremental, full._full


def _stream(text, rng):
    """Cumulative texts of `text` streamed in random-sized deltas (one char to a few dozen)."""
    cut, cuts = 0, []
    while cut < len(text):
        cut = min(len(text), cut + rng.randint(1, 24))
        cuts.append(text[:cut])
    return cuts


def _assert_same_deltas(text, rng):
    incremental, full = _pair()
    prefixes = _stream(text, rng)
    for i, prefix in enumerate(prefixes):
        finished = i == len(prefixes) - 1
        assert incremental.parse(prefix, finished=finished) == full.parse(prefix, finished=finished)
    assert incremental.finish_reason({}) == full.finish_reason({})


@pytest.mark.parametrize(
    "text",
    [
        "<think>\nreasoning about it</think>\n\nThe answer is 4.",
        "no think open, then</think>content after",
        "text before <think>late reasoning</think>content",
        "x</think>y<think>z still thinking",
        "<think>never closes",
        "<think>reasoning</think>content with <function=f>\n<parameter=a>1</parameter></function>",
        "",
    ],
)
def test_matches_full_parser_on_structures(text):
    for seed in range(20):
        _assert_same_deltas(text, random.Random(seed))


def test_matches_full_parser_on_random_texts():
    rng = random.Random(0)
    for _ in range(500):
        text = "".join(rng.choice(_PIECES) for _ in range(rng.randint(0, 60)))
        _assert_same_deltas(text, rng)


@pytest.mark.parametrize(
    "parsers, overrides",
    [
        (_PARSERS, dict(tools_requested=True)),
        (_PARSERS, dict(chat_template_kwargs={"enable_thinking": False})),
        (["qwen3-coder-tool"], {}),
        (["deepseek-r1-reasoning", "nemotron-v3-reasoning"], {}),
    ],
)
def test_full_parser_kept_where_not_equivalent(parsers, overrides):
    (parser,) = _streaming_chat_parsers(parsers, _options(**overrides), 1)
    assert isinstance(parser, StreamingChatParser)
