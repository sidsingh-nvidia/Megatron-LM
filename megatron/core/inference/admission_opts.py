# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Temporary switches for ablating request-admission host optimizations.

Experiment-only: each name in ENABLED turns on one optimization so a microbenchmark can
measure them one at a time in a single process. Read at call time, so callers may mutate
ENABLED between runs. Initialized from MCORE_ADMISSION_OPTS (comma-separated).
"""

import os

ENABLED = {name for name in os.environ.get("MCORE_ADMISSION_OPTS", "").split(",") if name}


def enabled(name: str) -> bool:
    """Whether optimization `name` is on."""
    return name in ENABLED
