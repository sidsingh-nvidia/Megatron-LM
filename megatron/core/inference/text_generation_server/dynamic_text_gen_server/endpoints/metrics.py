# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import os

logger = logging.getLogger(__name__)

# When running with num_replicas > 1, each replica is a separate process sharing
# the same socket. Set PROMETHEUS_MULTIPROC_DIR to a writable directory before
# starting the server so that all replicas contribute to the same metrics and the
# /metrics endpoint aggregates across them. Without it, only the replica that
# handles the scrape request reports its own counters.

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from quart import Blueprint, Response

_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

OUTPUT_TOKENS = Counter(
    "megatron_output_tokens_total",
    "Total completion tokens generated",
)
PROMPT_TOKENS = Counter(
    "megatron_prompt_tokens_total",
    "Total prompt tokens processed",
)
REQUESTS_TOTAL = Counter(
    "megatron_requests_total",
    "Total requests handled, by outcome",
    ["status"],  # "success" or "error"
)
IN_FLIGHT = Gauge(
    "megatron_requests_in_flight",
    "Requests currently waiting on the engine",
)
REQUEST_DURATION = Histogram(
    "megatron_request_duration_seconds",
    "Engine call wall-clock latency (excludes request parsing)",
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600],
)
OUTPUT_TPS = Histogram(
    "megatron_output_tokens_per_second",
    "Output tokens per second, per request",
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

bp = Blueprint("metrics_api", __name__)


@bp.route("/metrics", methods=["GET"])
async def metrics():
    if _MULTIPROC_DIR:
        from prometheus_client.multiprocess import MultiProcessCollector

        registry = CollectorRegistry()
        MultiProcessCollector(registry)
    else:
        registry = REGISTRY
    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)
