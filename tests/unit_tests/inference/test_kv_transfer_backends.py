# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.disaggregation.ssm_reshard import SSMShardLayout, SSMStateDims
from megatron.core.inference.disaggregation.transfer_backends import base


class _FakeCudaBuffer:
    is_cuda = True


@pytest.mark.parametrize(
    ("package_name", "distribution_name"), [("nixl_cu12", "nixl-cu12"), ("nixl_cu13", "nixl-cu13")]
)
def test_nixl_detects_active_cuda_distribution(monkeypatch, package_name, distribution_name):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    monkeypatch.setattr(
        nixl_mod.importlib_metadata,
        "packages_distributions",
        lambda: {"nixl": ["nixl"], package_name: [distribution_name]},
    )

    assert nixl_mod._detect_nixl_variant(f"{package_name}._api") == package_name


@pytest.mark.parametrize("configured_tls", ["tcp", "^cuda", "^cuda_copy,cuda_ipc,gdr_copy"])
def test_nixl_rejects_ucx_config_without_cuda_transport(monkeypatch, configured_tls):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    monkeypatch.setenv("UCX_TLS", configured_tls)

    with pytest.raises(RuntimeError, match="does not enable a CUDA transport"):
        nixl_mod._validate_ucx_transport_config(_FakeCudaBuffer())


@pytest.mark.parametrize("configured_tls", [None, "all", "rc,cuda_copy", "^tcp"])
def test_nixl_accepts_ucx_config_with_cuda_transport(monkeypatch, configured_tls):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    monkeypatch.delenv("UCX_MEMTYPE_CACHE", raising=False)
    if configured_tls is None:
        monkeypatch.delenv("UCX_TLS", raising=False)
    else:
        monkeypatch.setenv("UCX_TLS", configured_tls)

    nixl_mod._validate_ucx_transport_config(_FakeCudaBuffer())

    assert nixl_mod.os.environ["UCX_MEMTYPE_CACHE"] == "n"


def test_nixl_rejects_cuda_major_variant_mismatch(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    monkeypatch.setattr(nixl_mod, "_NIXL_VARIANT", "nixl_cu12")
    monkeypatch.setattr(nixl_mod.torch.version, "cuda", "13.2")

    with pytest.raises(RuntimeError, match="NIXL selected nixl_cu12"):
        nixl_mod._validate_nixl_cuda_support(object(), _FakeCudaBuffer())


def test_nixl_rejects_ucx_without_vram_support(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def get_backend_mem_types(self, backend):
            assert backend == "UCX"
            return ["DRAM"]

    monkeypatch.setattr(nixl_mod, "_NIXL_VARIANT", "nixl_cu13")
    monkeypatch.setattr(nixl_mod.torch.version, "cuda", "13.2")

    with pytest.raises(RuntimeError, match="does not report CUDA/VRAM"):
        nixl_mod._validate_nixl_cuda_support(FakeAgent(), _FakeCudaBuffer())


def test_nixl_accepts_matching_variant_with_vram_support(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def get_backend_mem_types(self, backend):
            assert backend == "UCX"
            return ["DRAM", "VRAM"]

    monkeypatch.setattr(nixl_mod, "_NIXL_VARIANT", "nixl_cu13")
    monkeypatch.setattr(nixl_mod.torch.version, "cuda", "13.2")

    nixl_mod._validate_nixl_cuda_support(FakeAgent(), _FakeCudaBuffer())


def test_backend_registry_selects_by_explicit_name():
    assert base.construct_kv_transfer_backend_class("nixl").name == "nixl"

    try:
        base.construct_kv_transfer_backend_class("unsupported")
    except ValueError as exc:
        assert "expected 'nixl'" in str(exc)
    else:
        raise AssertionError("unsupported backend should raise")


def test_ssm_geometry_uses_conv_and_recurrent_state_names():
    layout = SSMShardLayout(
        global_rank=0,
        tp_size=1,
        tp_rank=0,
        layer_start=0,
        num_layers=1,
        dims=SSMStateDims(nheads=2, headdim=4, d_state=5, ngroups=1, d_conv=3),
    )
    memory_buffer = torch.zeros(1, 3, 2, 4, 5)
    geometry = base.compute_buffer_geometry(
        memory_buffer,
        expected_num_blocks=3,
        backend_name="test",
        heads_per_partition=2,
        ssm_layout=layout,
        ssm_state_kind="recurrent",
    )

    metadata = base.export_geometry_meta(geometry, layout)
    assert metadata["ssm_layout"]["dims"]["nheads"] == 2
    assert "mamba_layout" not in metadata

    with pytest.raises(ValueError, match="'conv' or 'recurrent'"):
        base.compute_buffer_geometry(
            memory_buffer,
            expected_num_blocks=3,
            backend_name="test",
            heads_per_partition=2,
            ssm_layout=layout,
            ssm_state_kind="ssm",
        )


def test_ssm_geometry_uses_explicit_live_slot_axis():
    layout = SSMShardLayout(
        global_rank=0,
        tp_size=1,
        tp_rank=0,
        layer_start=0,
        num_layers=3,
        dims=SSMStateDims(nheads=2, headdim=4, d_state=5, ngroups=1, d_conv=3),
    )
    geometry = base.compute_buffer_geometry(
        torch.zeros(3, 3, 2, 4, 5),
        expected_num_blocks=3,
        backend_name="test",
        heads_per_partition=2,
        ssm_layout=layout,
        ssm_state_kind="recurrent",
    )

    assert geometry.blocks_axis == 1


def test_nixl_direct_backend_exports_metadata_with_fake_agent(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgentConfig:
        def __init__(self, *, enable_prog_thread):
            self.enable_prog_thread = enable_prog_thread

    class FakeAgent:
        def __init__(self, name, config):
            self.name = name
            self.config = config

        def get_agent_metadata(self):
            return b"agent-meta"

        def register_memory(self, tensor):
            return ("reg", tuple(tensor.shape))

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", FakeAgentConfig)

    backend = nixl_mod.NixlTransferBackend(
        "prefill", torch.zeros(2, 3, 5, dtype=torch.float32), expected_num_blocks=3
    )
    metadata = backend.export_meta()

    assert metadata["agent_name"] == "prefill"
    assert metadata["bytes_per_slice"] == 20
    assert metadata["num_outer"] == 2
    assert metadata["num_blocks"] == 3
    assert metadata["blocks_axis"] == 1
    assert backend._agent.config.enable_prog_thread is True


def test_nixl_registered_buffers_share_agent_and_peer_cache(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    agents = []

    class FakeAgent:
        def __init__(self, name, config):
            self.registrations = []
            self.deregistrations = []
            agents.append(self)

        def get_agent_metadata(self):
            return f"registrations={len(self.registrations)}".encode()

        def register_memory(self, tensor):
            self.registrations.append(tensor)
            return ("reg", len(self.registrations))

        def deregister_memory(self, registration):
            self.deregistrations.append(registration)

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", lambda **_: object())

    backend = nixl_mod.NixlTransferBackend(
        "prefill", torch.zeros(2, 3, 5, dtype=torch.float32), expected_num_blocks=3
    )
    sibling = backend.new_registered_buffer(
        agent_name="prefill-mamba-conv",
        memory_buffer=torch.zeros(2, 4, 5, dtype=torch.float32),
        expected_num_blocks=4,
    )

    assert len(agents) == 1
    assert sibling._agent is backend._agent
    assert sibling._known_peers is backend._known_peers
    assert sibling.export_meta()["agent_name"] == "prefill"
    assert backend.export_meta()["agent_metadata_b64"] == "cmVnaXN0cmF0aW9ucz0y"

    agent = backend._agent
    agent_context = backend._agent_context
    assert agent_context.ref_count == 2

    backend.close()
    assert agent_context.ref_count == 1
    assert agent_context.agent is agent
    assert sibling._agent is agent

    sibling.close()
    assert agent_context.ref_count == 0
    assert agent_context.agent is None
    assert agent.deregistrations == [("reg", 1), ("reg", 2)]


def test_nixl_begin_pull_blocks_uses_remote_metadata_with_fake_agent(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def __init__(self, name, config):
            self.name = name
            self.transferred = False

        def get_agent_metadata(self):
            return b"local"

        def register_memory(self, tensor):
            return ("reg", tuple(tensor.shape))

        def add_remote_agent(self, metadata):
            assert metadata == b"remote"
            return "peer"

        def get_xfer_descs(self, tuples, mem_type):
            assert mem_type == "VRAM"
            return tuples

        def initialize_xfer(self, op, local_desc, remote_desc, peer_id):
            assert op == "READ"
            assert peer_id == "peer"
            return (local_desc, remote_desc)

        def transfer(self, xfer):
            self.transferred = True

        def check_xfer_state(self, xfer):
            assert self.transferred
            return "DONE"

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", lambda **_: object())

    backend = nixl_mod.NixlTransferBackend(
        "decode", torch.zeros(2, 3, 5, dtype=torch.float32), expected_num_blocks=3
    )
    peer_meta = {
        "agent_name": "prefill",
        "agent_metadata_b64": "cmVtb3Rl",
        "base_addr": 1234,
        "bytes_per_slice": 20,
        "num_outer": 2,
        "outer_stride_bytes": 60,
        "num_blocks": 3,
        "device_id": 0,
        "blocks_axis": 1,
    }
    backend.begin_pull_blocks(peer_meta, [1], [2]).wait()

    assert backend._agent.transferred is True


def test_nixl_begin_pull_blocks_returns_pollable_handle(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def __init__(self, name, config):
            self.name = name
            self.transfers = 0
            self.polls = 0

        def get_agent_metadata(self):
            return b"local"

        def register_memory(self, tensor):
            return ("reg", tuple(tensor.shape))

        def add_remote_agent(self, metadata):
            assert metadata == b"remote"
            return "peer"

        def get_xfer_descs(self, tuples, mem_type):
            assert mem_type == "VRAM"
            return tuples

        def initialize_xfer(self, op, local_desc, remote_desc, peer_id):
            assert op == "READ"
            assert peer_id == "peer"
            return {"local": local_desc, "remote": remote_desc}

        def transfer(self, xfer):
            self.transfers += 1

        def check_xfer_state(self, xfer):
            self.polls += 1
            return "DONE" if self.polls >= 2 else "PENDING"

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", lambda **_: object())

    backend = nixl_mod.NixlTransferBackend(
        "decode", torch.zeros(2, 3, 5, dtype=torch.float32), expected_num_blocks=3
    )
    peer_meta = {
        "agent_name": "prefill",
        "agent_metadata_b64": "cmVtb3Rl",
        "base_addr": 1234,
        "bytes_per_slice": 20,
        "num_outer": 2,
        "outer_stride_bytes": 60,
        "num_blocks": 3,
        "device_id": 0,
        "blocks_axis": 1,
    }

    handle = backend.begin_pull_blocks(peer_meta, [1], [2])

    assert backend._agent.transfers == 1
    assert backend._agent.polls == 0
    assert handle.poll() is False
    assert handle.poll() is True


def _kv_geometry_kwargs(tp_size, tp_rank, num_kv_heads, heads_per_partition, rank):
    return dict(
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_kv_heads_global=num_kv_heads,
        heads_per_partition=heads_per_partition,
        head_dim=4,
        tokens_per_block=2,
        global_rank=rank,
        pp_size=1,
        pp_rank=0,
        num_layers_global=3,
        layer_start=0,
        layer_end=3,
    )


@pytest.mark.parametrize("tp_size", [4, 8])
def test_kv_geometry_accepts_replicated_kv_heads(tp_size):
    """2 KV heads at TP4/TP8: every rank holds one replicated head, as
    DynamicInferenceContext sizes it, and still gets a reshard layout."""
    buf = torch.zeros(2, 3, 5, 2, 1, 4)
    for tp_rank in range(tp_size):
        geometry = base.compute_buffer_geometry(
            buf,
            5,
            backend_name="test",
            **_kv_geometry_kwargs(tp_size, tp_rank, 2, 1, tp_rank),
        )
        assert geometry.layout is not None
        head = tp_rank // (tp_size // 2)
        assert geometry.layout.head_range() == (head, head + 1)


def test_kv_geometry_rejects_wrong_replicated_head_count():
    buf = torch.zeros(2, 3, 5, 2, 2, 4)
    geometry = base.compute_buffer_geometry(
        buf, 5, backend_name="test", **_kv_geometry_kwargs(4, 0, 2, 2, 0)
    )
    assert geometry.layout is None


def test_nixl_pull_replicated_kv_heads_reads_one_replica(monkeypatch):
    """Decode TP4 rank 1 pulling from a TP4 prefill with 2 replicated KV heads
    issues one whole-slice read from the matching prefill replica (rank 1)."""
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def __init__(self, name, config):
            self.name = name
            self.reads = []

        def get_agent_metadata(self):
            return self.name.encode()

        def register_memory(self, tensor):
            return ("reg", tuple(tensor.shape))

        def add_remote_agent(self, metadata):
            return metadata.decode()

        def get_xfer_descs(self, tuples, mem_type):
            return list(tuples)

        def initialize_xfer(self, op, local_desc, remote_desc, peer_id):
            assert op == "READ"
            self.reads.append((peer_id, local_desc, remote_desc))
            return peer_id

        def transfer(self, xfer):
            pass

        def check_xfer_state(self, xfer):
            return "DONE"

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", lambda **_: object())

    def make(role, tp_rank):
        return nixl_mod.NixlTransferBackend(
            f"{role}-rank{tp_rank}",
            torch.zeros(2, 3, 5, 2, 1, 4),
            expected_num_blocks=5,
            **_kv_geometry_kwargs(4, tp_rank, 2, 1, tp_rank),
        )

    prefill_metas = [make("prefill", r).export_meta() for r in range(4)]
    decode = make("decode", 1)
    decode.begin_pull_blocks({"tp_metas": prefill_metas}, [0, 3], [2, 4]).wait()

    assert [peer for peer, _, _ in decode._agent.reads] == ["prefill-rank1"]
    _, local_desc, remote_desc = decode._agent.reads[0]
    # Whole slices: 2 blocks x (2 K/V planes x 3 layers) outer slices.
    assert len(local_desc) == len(remote_desc) == 2 * 6
