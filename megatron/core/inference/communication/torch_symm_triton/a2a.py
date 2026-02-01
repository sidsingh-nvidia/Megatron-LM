# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock

import torch


import triton
import triton.language as tl


from .barrier import symm_mem_sync
from .multimem_asm import ld_128, st_128
from .utils import sync_threads

import torch.distributed._symmetric_memory as symm_mem
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist

import os


@triton.jit
def _all_to_all_kernel(
    buffer_ptrs_addr,    # Pointers to remote input buffers (Symmetric Memory)
    output_ptr,          # Local output buffer
    signal_pad_ptrs,     # Synchronization pads for barrier
    tokens_per_peer: tl.constexpr,
    hidden_size: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, # Next power of 2 of (hidden_size // 8)
):
    # 1. Sync: Wait for all ranks to be ready.
    # Note: Setting hasSubsequentMemAccess=True acts as an 'acquire' fence 
    # to ensure remote data is visible before we start the 'Pull'.
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=False, 
    )
    sync_threads()

    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)
    b_ptrs = buffer_ptrs_addr.to(tl.pointer_type(tl.uint64))
    
    # 2. Pre-calculate the mask for the hidden dimension chunks
    # We move 8 elements (16 bytes) per chunk.
    chunk_offsets = tl.arange(0, BLOCK_SIZE)
    # Mask out chunks that would start beyond the hidden_size
    mask = (chunk_offsets * 8) < hidden_size

    # 3. Iterate through peers using a rotation to spread NVLink traffic
    for peer_offset in range(WORLD_SIZE):
        peer_rank = (RANK + peer_offset) % WORLD_SIZE
        
        # Calculate base offsets for the peer-to-peer chunk
        remote_el_offset = RANK * tokens_per_peer * hidden_size
        local_el_offset = peer_rank * tokens_per_peer * hidden_size
        
        t = pid
        while t < tokens_per_peer:
            # Calculate byte addresses for the current token (bf16 = 2 bytes)
            remote_byte_start = (remote_el_offset + (t * hidden_size)) * 2
            local_byte_start  = (local_el_offset + (t * hidden_size)) * 2

            # 128-bit = 16 bytes per chunk
            byte_offsets = chunk_offsets * 16
            
            # A. Load Remote Base Address
            peer_ptr_raw = tl.load(b_ptrs + peer_rank)
            
            # B. Vectorized Load (Pull from Peer over NVLink)
            # The mask predicates the PTX ld.global.v4.u32 instruction
            src_ptr = peer_ptr_raw + remote_byte_start + byte_offsets
            r0, r1, r2, r3 = ld_128(
                src_ptr, 
                mask=mask, 
                multicast_op=False
            )

            # C. Vectorized Store (Local VRAM)
            dst_ptr = output_ptr + local_byte_start + byte_offsets
            st_128(
                dst_ptr, r0, r1, r2, r3, 
                mask=mask, 
                multicast_op=False
            )
            
            t += num_progs

def multimem_all_to_all(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    **kwargs,
) -> torch.Tensor:
    assert output_tensor.shape == input_tensor.shape, \
        "Output tensor shape must match input tensor shape."
    assert output_tensor.ndim == 2 
    assert output_tensor.dtype == torch.bfloat16
    assert input_tensor.dtype == torch.bfloat16

    num_tokens = input_tensor.shape[0]
    hidden_size = input_tensor.shape[-1]
    
    # 128-bit chunks cover 8 bf16 elements
    num_chunks = (hidden_size + 7) // 8
    # Force BLOCK_SIZE to next power of 2 as required by Triton
    block_size = triton.next_power_of_2(num_chunks)
    
    config = {
        "max_num_blocks": kwargs.get("max_num_blocks", 108), # Saturate Blackwell SMs
        "num_warps": kwargs.get("num_warps", 32),
    }

    world_size = symm_mem_hdl.world_size
    assert num_tokens % world_size == 0, \
        "Number of tokens must be divisible by world size."
    tokens_per_peer = (num_tokens // world_size)

    # Grid logic
    # 1 block handles 1 token
    num_blocks = min(config["max_num_blocks"], tokens_per_peer)

    _all_to_all_kernel[(num_blocks, 1, 1)](
        buffer_ptrs_addr=symm_mem_hdl.buffer_ptrs_dev,
        output_ptr=output_tensor.data_ptr(),
        signal_pad_ptrs=symm_mem_hdl.signal_pad_ptrs_dev,
        tokens_per_peer=tokens_per_peer,
        hidden_size=hidden_size,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=world_size,
        BLOCK_SIZE=block_size,
        num_warps=config["num_warps"],
    )

    return output_tensor

def benchmark_with_cuda_graphs(output_buffer, input_buffer_ep, input_hdl_ep, iterations=100):
    rank = dist.get_rank()
    
    # 1. Warmup (Required for Triton JIT and Autotuning)
    for _ in range(5):
        multimem_all_to_all(
            output_tensor=output_buffer,
            input_tensor=input_buffer_ep,
            symm_mem_hdl=input_hdl_ep,
        )
    torch.cuda.synchronize()

    # 2. Graph Capture
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    
    # Static inputs for the graph
    # Note: The memory addresses of output_buffer and input_buffer must not change!
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(iterations):
                multimem_all_to_all(
                    output_tensor=output_buffer,
                    input_tensor=input_buffer_ep,
                    symm_mem_hdl=input_hdl_ep,
                )
    torch.cuda.current_stream().wait_stream(s)    

    # 3. Timing Loop
    dist.barrier() # Global sync before starting the clock
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push("Torch Symm")
    start_event.record()
    g.replay()
    end_event.record()
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.synchronize()
    
    # 4. Results
    avg_latency_ms = start_event.elapsed_time(end_event) / iterations
    
    if rank == 0:
        # Bandwidth calculation
        # total_bytes = total_tokens * hidden_size * 2 (for bf16)
        total_tokens = input_buffer_ep.shape[0]
        hsize = input_buffer_ep.shape[1]
        total_gb = (total_tokens * hsize * 2) / 1e9
        gbps = total_gb / (avg_latency_ms / 1000)
        
        print(f"--- Torch Symm CUDA Graph Performance ---")
        print(f"Num tokens: {total_tokens}, Hidden Size: {hsize}")
        print(f"Avg Latency: {avg_latency_ms * 1000:.4f} us")
        print(f"Effective Bandwidth: {gbps:.2f} GB/s per rank")

def benchmark_nccl_all_to_all(num_tokens, hsize, iterations=100):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    # 1. Setup Buffers (Standard VRAM, not symmetric)
    input_tensor = torch.empty(num_tokens, hsize, dtype=torch.bfloat16, device=device)
    output_tensor = torch.empty_like(input_tensor)
    input_tensor.fill_(rank + 1.0)

    # Define equal splits for All-to-All
    # NCCL expects a list of integers representing the number of elements in each chunk
    split_size = (num_tokens // world_size)
    send_splits = [split_size] * world_size
    recv_splits = [split_size] * world_size

    # 2. Warmup
    for _ in range(5):
        dist.all_to_all_single(output_tensor, input_tensor, 
                               output_split_sizes=recv_splits, 
                               input_split_sizes=send_splits)
    torch.cuda.synchronize()

    # 3. CUDA Graph Capture
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(iterations):
                dist.all_to_all_single(output_tensor, input_tensor, 
                                    output_split_sizes=recv_splits, 
                                    input_split_sizes=send_splits)
    torch.cuda.current_stream().wait_stream(s)

    # 4. Benchmark Loop
    dist.barrier()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push("NCCL")
    start_event.record()
    g.replay()
    end_event.record()
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.synchronize()
    
    # 5. Report
    avg_latency_ms = start_event.elapsed_time(end_event) / iterations
    if rank == 0:
        total_gb = (num_tokens * hsize * 2) / 1e9
        gbps = total_gb / (avg_latency_ms / 1000)
        print(f"--- NCCL (CUDA Graph) Performance ---")
        print(f"Avg Latency: {avg_latency_ms * 1000:.4f} us")
        print(f"Effective Bandwidth: {gbps:.2f} GB/s per rank")

    return avg_latency_ms


if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)

    tp_size = 1 
    ep_size = dist.get_world_size() // tp_size

    # create tp and ep groups 
    for i in range(ep_size):
        ranks = [j for j in range(dist.get_world_size()) if j // tp_size == i]
        group = dist.new_group(ranks=ranks, backend='nccl')
        if rank in ranks:
            tp_group = group 

    for i in range(tp_size):
        ranks = [j for j in range(dist.get_world_size()) if j % tp_size == i]
        group = dist.new_group(ranks=ranks, backend='nccl')
        if rank in ranks:
            ep_group = group 

    symm_mem.enable_symm_mem_for_group(ep_group.group_name)
    symm_mem.enable_symm_mem_for_group(tp_group.group_name)
    # Create input tensor in symmetric memory
    num_tokens, hsize = 512, 5120
    
    input_buffer_ep = symm_mem.empty(num_tokens, hsize, dtype=torch.bfloat16, device='cuda')
    input_hdl_ep = symm_mem.rendezvous(input_buffer_ep, ep_group)
    output_buffer = torch.empty_like(input_buffer_ep)

    input_buffer_ep.fill_(rank + 1)
    
    input_buffer_tp = symm_mem.empty(num_tokens, hsize, dtype=torch.bfloat16, device='cuda')
    input_hdl_tp = symm_mem.rendezvous(input_buffer_tp, tp_group)
    input_hdl_tp = symm_mem.rendezvous(input_buffer_tp, tp_group)

    input_buffer_tp.fill_(rank + 1)

    
    
    multimem_all_to_all(
        output_tensor=output_buffer,
        input_tensor=input_buffer_ep,
        symm_mem_hdl=input_hdl_ep,
    )
    
    
    if rank == 0:
        print(f"Output Buffer after All-to-All | rank = {rank}:")
        print(output_buffer)

    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStart()

    # Replace your range(10) loop in __main__ with this:
    if rank == 0:
        print("Starting CUDA Graph Benchmark...")
    
   
    benchmark_with_cuda_graphs(output_buffer, input_buffer_ep, input_hdl_ep)
   
    benchmark_nccl_all_to_all(num_tokens, hsize)
   
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()



        