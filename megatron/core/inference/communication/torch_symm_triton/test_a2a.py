import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed._symmetric_memory as symm_mem
from a2a import multimem_all_to_all # Import your kernel wrapper

def run_test(num_tokens, hidden_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 1. Setup Symmetric Memory
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    input_buffer = symm_mem.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')
    input_hdl = symm_mem.rendezvous(input_buffer, dist.group.WORLD)
    
    # 2. Fill with identifiable data: (rank + 1) + epsilon_offsets
    # This helps catch if we accidentally pull from the wrong peer or wrong offset
    local_data = (torch.ones((num_tokens, hidden_size), device='cuda', dtype=torch.bfloat16) * (rank + 1))
    # Add a small gradient so every element is unique
    offsets = torch.linspace(0, 0.1, steps=hidden_size, device='cuda', dtype=torch.bfloat16)
    input_buffer.copy_(local_data + offsets)
    
    # 3. Reference Result (NCCL)
    expected_output = torch.empty_like(input_buffer)
    dist.all_to_all_single(expected_output, input_buffer)
    
    # 4. Triton Implementation Result
    actual_output = torch.zeros_like(input_buffer)
    torch.cuda.synchronize()
    
    # Run the kernel
    multimem_all_to_all(
        output_tensor=actual_output,
        input_tensor=input_buffer,
        symm_mem_hdl=input_hdl,
        max_num_blocks=160, # Use Blackwell SM count for test
    )
    torch.cuda.synchronize()

    # 5. Verification
    # Check for NaNs/Infs first
    assert not torch.isnan(actual_output).any(), f"Rank {rank}: Output contains NaNs"
    
    # Strict equality check for bf16
    if torch.allclose(actual_output, expected_output, atol=1e-3, rtol=1e-3):
        print(f"✅ Rank {rank}: Correctness Verified!")
    else:
        # Debugging: Find the first mismatch
        diff = (actual_output - expected_output).abs()
        max_diff = diff.max()
        print(f"❌ Rank {rank}: Mismatch detected! Max diff: {max_diff} | tokens = {num_tokens}, hsize = {hidden_size}")
        
    

if __name__ == "__main__":
    NUM_TOKENS = 128
    HIDDEN_SIZE = 1024 # Test with your MoE h_size
    dist.init_process_group()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    for NUM_TOKENS in [32, 128, 512, 40, 20]:
        for HIDDEN_SIZE in [1024, 5120, 4096, 1000]:
            run_test(NUM_TOKENS, HIDDEN_SIZE)


    dist.destroy_process_group()

    