import torch
from typing import List, Optional

def permute_tokens(tensor: torch.Tensor, mappings: torch.Tensor) -> torch.Tensor:
    # mappings can be on CPU or GPU
    permuted = tensor[mappings]
    assert permuted.is_contiguous()
    return permuted