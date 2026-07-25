# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


try:
    from .metrics import bp as Metrics

    _has_metrics = True
except ImportError:
    _has_metrics = False

try:
    from .chat_completions import bp as ChatCompletions
    from .completions import bp as Completions
    from .health import bp as Health
    from .profile import bp as Profile

    __all__ = [Completions, ChatCompletions, Health, Profile]
    if _has_metrics:
        __all__.append(Metrics)
except ImportError:
    __all__ = []
