import torch

import triton
import triton.language as tl
import numpy as np
from typing import Tuple, List, Union

# @triton.autotune(configs=[
#     triton.Config(kwargs={"BLOCK_SIZE": 64}, num_warps=2),
    
#     triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=2),
#     triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=4),
    
#     triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=4),
#     triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=8),
    
#     triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=4),
#     triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=8),
#     triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=16),
    
#     triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),
#     triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),
#     triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),
#     triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=32),
# ], key=["hidden_size"],
# )
@triton.jit
def _permute_tokens_kernel(
    out_ptr, # buffer for permuted tokens 
    in_ptr, # input tokens
    mapping,
    hidden_size, # int
    BLOCK_SIZE: tl.constexpr # division of hidden_size, should be tuned (default 128)
):
    
    token_id = tl.program_id(axis=0)
    block_id = tl.program_id(axis=1)

    target_pos = tl.load(mapping + token_id)

    src_start = token_id * hidden_size + block_id * BLOCK_SIZE
    src_offsets = src_start + tl.arange(0, BLOCK_SIZE)
    src_data = tl.load(in_ptr + src_offsets)
    
    target_start = target_pos * hidden_size + block_id * BLOCK_SIZE
    target_offsets = target_start + tl.arange(0, BLOCK_SIZE)
    
    tl.store(out_ptr + target_offsets, src_data)

def permute_tokens(tokens: torch.Tensor, 
                   mappings: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    # permute tokens according to its expert id
    assert len(tokens.shape) == 2 # [num_tokens, hidden_size]
    num_tokens, hiddens_size = tokens.shape
    assert(tokens.is_contiguous())
    permuted_tokens = torch.empty((num_tokens, hiddens_size), device=tokens.device, dtype=tokens.dtype)
    
    if not torch.is_tensor(mappings):
        mappings = torch.tensor(mappings, dtype=torch.int32, device=tokens.device)
    
    grid = lambda META: (num_tokens, triton.cdiv(hiddens_size, META["BLOCK_SIZE"]))    
    _permute_tokens_kernel[grid](
        permuted_tokens, 
        tokens, 
        mappings.to(tokens.device),
        hiddens_size,
        BLOCK_SIZE=1024
    )
    return permuted_tokens

def get_mappings_from_exp_ids_torch(exp_ids: torch.Tensor, num_experts: int):
    assert len(exp_ids.shape) == 1 # [num_tokens]
    
    exp_ids = exp_ids.to("cpu")
    exp_cnt = torch.bincount(exp_ids, minlength=num_experts)
    exp_cumsum = torch.cumsum(exp_cnt, dim=0)
    
    mappings = torch.empty(exp_ids.shape[0], device="cpu", dtype=torch.int64)  
    
    for i, id in enumerate(exp_ids):
        exp_cumsum[id] -= 1
        mappings[i] = exp_cumsum[id]
        
    return mappings, exp_cnt

def get_mappings_from_exp_ids_cuda(exp_ids: torch.Tensor, num_experts: int):
    assert len(exp_ids.shape) == 1 # [num_tokens]
    
    # exp_ids = exp_ids.to("cpu")
    exp_cnt = torch.bincount(exp_ids, minlength=num_experts)
    exp_cumsum = torch.cumsum(exp_cnt, dim=0)
    
    exp_cumsum = exp_cumsum.cpu()
    mappings = torch.empty(exp_ids.shape[0], device="cpu", dtype=torch.int64)  
    
    for i, id in enumerate(exp_ids):
        exp_cumsum[id] -= 1
        mappings[i] = exp_cumsum[id]
        
    return mappings, exp_cnt

def get_mappings_from_exp_ids_py(exp_ids: torch.Tensor, num_experts: int) -> Tuple[List[int], List[int]]:
    assert len(exp_ids.shape) == 1 # [num_tokens]
    
    exp_ids = exp_ids.view(-1).tolist()
    
    exp_cnt = [0] * num_experts
    exp_cumsum = [0] * num_experts
    
    for id in exp_ids:
        exp_cnt[id] += 1
        
    exp_cumsum[0] = exp_cnt[0]
    for i in range(1, num_experts):
        exp_cumsum[i] = exp_cumsum[i-1] + exp_cnt[i]
    # exp_cnt = torch.bincount(exp_ids, minlength=num_experts)
    # exp_cumsum = torch.cumsum(exp_cnt, dim=0)
    
    # mappings = torch.empty(exp_ids.shape[0], device="cpu", dtype=torch.int64)  
    mappings = [0] * len(exp_ids)
    
    for i, id in enumerate(exp_ids):
        exp_cumsum[id] -= 1
        mappings[i] = exp_cumsum[id]
        
    return mappings, exp_cnt

def get_mappings_from_exp_ids_numpy(exp_ids: torch.Tensor, num_experts: int):
    assert len(exp_ids.shape) == 1 # [num_tokens]
    
    exp_ids = exp_ids.view(-1).cpu().numpy()
    
    exp_cnt = np.zeros(num_experts, dtype=np.int32)
    exp_cumsum = np.zeros(num_experts, dtype=np.int32)
    
    for id in exp_ids:
        exp_cnt[id] += 1
        
    exp_cumsum[0] = exp_cnt[0]
    for i in range(1, num_experts):
        exp_cumsum[i] = exp_cumsum[i-1] + exp_cnt[i]
    # exp_cnt = torch.bincount(exp_ids, minlength=num_experts)
    # exp_cumsum = torch.cumsum(exp_cnt, dim=0)
    
    # mappings = torch.empty(exp_ids.shape[0], device="cpu", dtype=torch.int64)  
    mappings = np.zeros(exp_ids.shape[0], dtype=np.int64)
    
    for i, id in enumerate(exp_ids):
        exp_cumsum[id] -= 1
        mappings[i] = exp_cumsum[id]
        
    return torch.from_numpy(mappings), exp_cnt