"""vLLM general plugin: TP-sharded shared expert for Nemotron-H under sequence-parallel MoE.

With ``--enable-expert-parallel``, TP > 1 and DP > 1 (``use_sequence_parallel_moe``), stock
vLLM gives every TP rank 1/TP of the tokens for the whole MoE block and therefore replicates
the shared expert (``disable_tp=True``): every rank reads the full shared-expert weights to
process 1/TP of the tokens. Decode is weight-bandwidth bound, so that costs TP x the shared
expert's weight reads, and TP x its memory.

Every TP rank already holds all the tokens before ``sequence_parallel_chunk``, so the shared
expert can stay TP-sharded on the full token set instead:

    shared  = shared_experts(x)                  # all tokens, TP-sharded, partial sum
    routed  = experts(sequence_parallel_chunk(x)) # 1/TP of the tokens, as before
    out     = all_reduce(place(routed) + shared)  # replaces the existing all-gather

``place`` zero-pads the rank's routed chunk into its rows of a full-size buffer, so the one
all-reduce both sums the shared expert's TP partials and gathers the routed chunks.

Put this directory on PYTHONPATH (its ``.dist-info`` registers the ``vllm.general_plugins``
entry point) and set ``VLLM_SP_SHARED_TP=1``. Without it, or when sequence-parallel MoE is
off, the model is unchanged.
"""

import copy
import os


def _patch_nemotron_h() -> None:
    import torch.nn.functional as F

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
    )
    from vllm.logger import init_logger
    from vllm.model_executor.models import nemotron_h
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    # Under the "vllm" logger, which vLLM configures; a top-level name would drop INFO.
    logger = init_logger(f"vllm.{__name__}")
    moe_cls = nemotron_h.NemotronHMoE
    if getattr(moe_cls, "_sp_shared_tp_patched", False):
        return
    orig_init = moe_cls.__init__
    orig_forward = moe_cls.forward

    def __init__(self, config, quant_config=None, parallel_config=None, prefix=""):
        sp = parallel_config is not None and parallel_config.use_sequence_parallel_moe
        n_shared = config.n_shared_experts or 0
        if not (sp and n_shared > 0):
            orig_init(self, config, quant_config, parallel_config, prefix)
            self._sp_shared_tp = False
            return
        # Build the routed experts without a shared expert, so the MoE runner neither runs nor
        # replicates it, then add a TP-sharded one that forward() runs on all the tokens.
        no_shared = copy.copy(config)
        no_shared.n_shared_experts = 0
        orig_init(self, no_shared, quant_config, parallel_config, prefix)
        self.n_shared_experts = n_shared
        self.shared_experts = nemotron_h.NemotronHMLP(
            config=config,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_shared_expert_intermediate_size * n_shared,
            quant_config=quant_config,
            reduce_results=False,
            is_sequence_parallel=False,
            prefix=f"{prefix}.shared_experts",
        )
        self._sp_shared_tp = True
        logger.info_once(
            "sp_shared_tp plugin: Nemotron-H shared expert TP-sharded over %d ranks under "
            "sequence-parallel MoE (one all-reduce replaces the MoE output all-gather).",
            get_tensor_model_parallel_world_size(),
        )

    def forward(self, hidden_states):
        if not self._sp_shared_tp:
            return orig_forward(self, hidden_states)
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = self.shared_experts(hidden_states)

        chunk = sequence_parallel_chunk(hidden_states)
        router_logits, _ = self.gate(chunk)
        routed_output = self.experts(hidden_states=chunk, router_logits=router_logits)

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        chunk_len = chunk.shape[0]
        routed_full = F.pad(
            routed_output, (0, 0, tp_rank * chunk_len, (tp_size - 1 - tp_rank) * chunk_len)
        )
        shared_full = F.pad(shared_output, (0, 0, 0, tp_size * chunk_len - num_tokens))
        output = tensor_model_parallel_all_reduce(routed_full + shared_full)
        return output[:num_tokens].view(num_tokens, hidden_dim)

    moe_cls.__init__ = __init__
    moe_cls.forward = forward
    moe_cls._sp_shared_tp_patched = True


def register() -> None:
    if os.environ.get("VLLM_SP_SHARED_TP", "0") != "1":
        return
    _patch_nemotron_h()
