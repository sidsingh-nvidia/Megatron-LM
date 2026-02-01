# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference-optimized AlltoAll Token Dispatcher with GPU-resident metadata.

This implementation keeps tokens_per_expert GPU-resident to enable use of
torch._grouped_mm without host synchronization.
"""

import torch
from typing import List, Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from megatron.core.transformer.moe.moe_utils import permute

import logging

class SymmetricMoEWorkspace:
    """
    A persistent workspace for Blackwell MoE collectives.
    Uses Symmetric Memory (NVSHMEM backend) for zero-copy-style pulls.
    """
    def __init__(
        self, 
        max_tokens_per_rank: int, 
        hidden_size: int, 
        num_experts: int, 
        ep_group: dist.ProcessGroup, 
        dtype: torch.dtype = torch.bfloat16
    ):
        self.device = torch.cuda.current_device()
        self.num_experts = num_experts
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(group=ep_group)
        self.rank = dist.get_rank(group=ep_group)
        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        
        # nsplits is total experts across the EP group because 
        # the collective takes per-expert input splits.
        self.nsplits = num_experts 

        # 1. Dispatch (Shuffle) Buffers
        # The Rank-Major buffer we fill before the All-to-All
        self.dispatch_inp = symm_mem.empty(
            (max_tokens_per_rank, hidden_size), dtype=dtype, device=self.device
        )
        
        # The Expert-Major buffer populated by the All-to-All
        # Worst case: one rank receives all tokens from all peers
        self.max_out_tokens = max_tokens_per_rank * self.ep_size 
        self.dispatch_out = symm_mem.empty(
            (self.max_out_tokens, hidden_size), dtype=dtype, device=self.device
        )

        # 2. Metadata Buffers
        # Holds the per-expert counts we provide to the collective
        self.in_splits = symm_mem.empty(
            (self.nsplits,), dtype=torch.int64, device=self.device
        )
        
        # Populated by the collective: Row 0 = counts, Row 1 = offsets
        # This metadata describes the Expert-Major layout of dispatch_out
        self.dispatch_metadata = symm_mem.empty(
            (2, self.nsplits), dtype=torch.int64, device=self.device
        )

        # 3. Combine (Un-shuffle) Buffers
        # Where tokens land after being pulled back to their original ranks
        self.combine_out = symm_mem.empty(
            (max_tokens_per_rank, hidden_size), dtype=dtype, device=self.device
        )
        
        # Metadata populated during the reverse (combine) operation
        self.combine_metadata = symm_mem.empty(
            (2, self.nsplits), dtype=torch.int64, device=self.device
        )

    def get_tokens_per_expert(self):
        """Returns the [num_experts] GPU tensor of token counts after dispatch."""
        return self.dispatch_metadata[0]

    def get_grouped_mm_offsets(self):
        """Returns the prefix-sum offsets required for GroupedMLP."""
        # result[i] = end index of expert i's tokens in dispatch_out
        return self.get_tokens_per_expert().cumsum(0).to(torch.int32)

    def set_input_splits(self, input_splits: torch.Tensor):
        """Registers the input_splits tensor for the dispatch collective."""
        self.in_splits.copy_(input_splits)

class InferenceAlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):
    """
    Inference-optimized AlltoAll token dispatcher.

    Key optimization: Returns tokens_per_expert as a GPU tensor (not moved to CPU)
    to enable torch._grouped_mm without host synchronization.

    Assumes tp_size == 1 (no tensor parallelism within experts).
    """
    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """
        Initialize the inference AlltoAll token dispatcher.

        Args are identical to MoEAlltoAllTokenDispatcher for compatibility.
        """
        super().__init__(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
        )
        self.use_nvshmem = False

    def set_nvshmem_usage(self, set_to: bool):
        self.use_nvshmem = set_to

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Preprocess routing map, ensuring tokens_per_expert is created on GPU.
        """
        if self.use_nvshmem:
            # only two states needed 
            # 1. input splits of local tokens - [global_num_experts]
            #    entry i = number of tokens for expert i on this rank
            # 2. num_out_tokens - total number of output tokens after dispatch

            # Only support dropless routing - keep things simple
            assert self.config.moe_expert_capacity_factor is None
            # Do not support this quantization feature yet. 
            assert not self.config.moe_router_padding_for_quantization

            num_local_tokens_per_expert = routing_map.sum(dim=0).long()
            if self.ep_size > 1:
                self.symmetric_workspace.set_input_splits(num_local_tokens_per_expert.view(-1))
                logging.info("Initialized symmetric workspace input_splits on GPU.")
            self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk
        else:
            return super().preprocess(routing_map)

    def dispatch_preprocess(self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor):
        """Prepares hidden states and probabilities for dispatch.

        This method reshapes the hidden states, computes communication metadata,
        and permutes the tokens and probabilities before the All-to-All communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            routing_map (torch.Tensor): The mapping of tokens to experts.
            probs (torch.Tensor): Routing probabilities.

        Returns:
            A tuple of permuted hidden states and probabilities.
        """
        if self.use_nvshmem:
            # Step 1. - Record input splits.
            self.preprocess(routing_map)
            # Step 2 - Permute tokens and probabilities by expert assignment.
            assert self.config.moe_permute_fusion 
            assert not self.drop_and_pad, "Drop-and-pad is not supported with nvshmem dispatch."
            self.hidden_shape_before_permute = hidden_states.shape
            (
                permutated_local_input_tokens,
                permuted_probs,
                self.reversed_local_input_permutation_mapping,
            ) = permute(
                    hidden_states,
                    routing_map,
                    probs=probs,
                    num_out_tokens=self.num_out_tokens,
                    fused=True,
                    drop_and_pad=False,
                )
            logging.info("permuted tokens")
            exit()
            return permutated_local_input_tokens, permuted_probs
        else:
            return super().dispatch_preprocess(
                hidden_states, routing_map, probs
            )
        
    def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
        """
        Perform all-to-all communication for dispatching tokens.

        This method performs the all-to-all communication step to dispatch tokens across
        expert parallel ranks. It synchronizes metadata at the appropriate point before
        performing the communication.

        Args:
            permutated_local_input_tokens (torch.Tensor): Pre-permuted input tokens.
            permuted_probs (torch.Tensor): Pre-permuted probabilities.

        Returns:
            A tuple of tokens and probabilities after All-to-All.
        """
        # Perform expert parallel AlltoAll communication
        # global_input_tokens = all_to_all(
        #     self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
        # )
        # global_probs = all_to_all(
        #     self.ep_group, permuted_probs, self.output_splits, self.input_splits
        # )
        inp = self.symmetric_workspace.dispatch_inp
        out = self.symmetric_workspace.dispatch_out
        in_splits = self.symmetric_workspace.in_splits
        out_splits_offsets = self.symmetric_workspace.dispatch_metadata
        group_name = self.ep_group.group_name
        align = 1

        torch.ops.symm_mem.all_to_all_vdev_2d(
            inp, out, in_splits, out_splits_offsets, group_name, major_align=align
        )

        logging.info("Completed AlltoAll dispatch collective using symmetric memory.")
        if dist.get_rank() == 0:
            print(out_splits_offsets[0])
            print(out_splits_offsets[1])
        exit()

        return global_input_tokens, global_probs

    def _maybe_dtoh_and_synchronize(
        self, point: str, tokens_per_expert: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """No-op for single GPU inference - all metadata stays on GPU.

        For single GPU (ep_size=1, tp_size=1):
        - input_splits, output_splits, output_splits_tp are all None (no AlltoAll needed)
        - tokens_per_expert stays on GPU for torch._grouped_mm
        - No DtoH transfers or synchronization required

        This enables fully CUDA-graphable MoE forward pass.
        """
        # Validate single GPU assumptions
        # assert self.ep_size == 1, (
        #     f"InferenceAlltoAllTokenDispatcher requires ep_size=1, got {self.ep_size}"
        # )
        assert self.tp_size == 1, (
            f"InferenceAlltoAllTokenDispatcher requires tp_size=1, got {self.tp_size}"
        )
        # assert self.input_splits is None, (
        #     "input_splits should be None for single GPU inference"
        # )
        # assert self.output_splits is None, (
        #     "output_splits should be None for single GPU inference"
        # )
        # assert self.output_splits_tp is None, (
        #     "output_splits_tp should be None for single GPU inference"
        # )
        assert not isinstance(self.num_out_tokens, torch.Tensor), (
            "num_out_tokens should be a Python int for dropless single GPU inference, "
            f"got {type(self.num_out_tokens)}. Ensure moe_expert_capacity_factor is None "
            "and moe_router_padding_for_quantization is False."
        )
        assert tokens_per_expert.is_cuda, (
            "tokens_per_expert should be on GPU for single GPU inference"
        )


        # No DtoH transfers needed - return tokens_per_expert unchanged (stays on GPU!)
        return tokens_per_expert

