import memmove_c
import torch

def permute_tokens(tensor: torch.Tensor, mappings: torch.Tensor) -> torch.Tensor:
    permuted = memmove_c.permute_tokens_cuda(tensor, mappings.cuda())
    return permuted