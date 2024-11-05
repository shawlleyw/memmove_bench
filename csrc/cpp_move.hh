#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor permute_tokens_cpp(torch::Tensor tokens, torch::Tensor mappings);