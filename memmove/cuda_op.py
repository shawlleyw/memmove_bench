import memmove_c
import torch
from typing import Union, List

def permute_tokens(tensor: torch.Tensor, mappings: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    if not torch.is_tensor(mappings):
        mappings = torch.tensor(mappings, dtype=torch.int32, device=tensor.device)
    
    permuted = memmove_c.permute_tokens_cuda(tensor, mappings.to(tensor.device))
    return permuted