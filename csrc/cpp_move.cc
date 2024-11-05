
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <assert.h>
#include <cstring>

#include "cpp_move.hh"

torch::Tensor permute_tokens_cpp(torch::Tensor tokens, torch::Tensor mappings) {
    // tokens are on GPU, mappings are on CPU
    assert(mappings.device().is_cpu());
    assert(tokens.device().is_cuda());
    assert(tokens.dtype() == torch::kBFloat16);
    c10::InferenceMode guard(true);
    auto tokens_cpu = tokens.to(torch::kCPU);
    auto moved_tokens = torch::empty_like(tokens_cpu);
    
    int batch = tokens_cpu.size(0);
    int hiddens = tokens_cpu.size(1);

    const int elem_size = sizeof(tokens.dtype());
    const int token_size = elem_size * hiddens;

    auto mappings_acc = mappings.accessor<int,1>();

    for (int i = 0; i < batch; i++) {
        void* src = (void*)tokens_cpu.data_ptr() + i * token_size;
        void* dest = (void*)moved_tokens.data_ptr() + mappings_acc[i] * token_size;
        std::memmove(dest, src, token_size);
    }

    return moved_tokens.to(torch::kCUDA);
}