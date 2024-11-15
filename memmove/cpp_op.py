import memmove_c
import torch
from typing import Union, List

def permute_tokens(tensor: torch.Tensor, mappings: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    
    if not torch.is_tensor(mappings):
        mappings = torch.tensor(mappings, dtype=torch.int32, device="cpu")
    
    permuted = memmove_c.permute_tokens_cpp(tensor, mappings.to("cpu"))
    assert permuted.is_cuda
    return permuted