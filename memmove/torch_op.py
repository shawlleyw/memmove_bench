import torch
from typing import List, Optional

def permute_tokens(tensor: torch.Tensor, mappings: torch.Tensor) -> torch.Tensor:
    # mappings can be on CPU or GPU
    permuted = tensor[mappings]
    assert permuted.is_contiguous()
    return permuted

def gpu_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.is_cuda
    return tensor.cpu()

def cpu_to_gpu(tensor: torch.Tensor) -> torch.Tensor:
    assert not tensor.is_cuda
    return tensor.cuda()