#include <cuda_runtime.h>
#include <iostream>

#include <torch/torch.h>
#include <torch/extension.h>
#include <assert.h>
#include <cstring>
#include <string>

#include "cuda_move.h"

template <class T>
__global__ void permute_tokens_kernel(T *d_out, T *d_in, long *mappings, int hidden_size, int chunk_size) {
    int token_id = blockIdx.x;
    int chunk_id = blockIdx.y;

    int tid = threadIdx.x;

    int p = mappings[token_id];
    T *dest = d_out + p * hidden_size;

    #pragma unroll
    for (int i = 0; i < chunk_size; i += blockDim.x) {
        dest[chunk_id * chunk_size + tid + i] = d_in[chunk_id * chunk_size + tid + i];
    }
}

template <class T>
void _permute_tokens_cuda(T *dest, T *src, long *mappings, int num_tokens, int hidden_size) {

    int chunk_size = 256;
    if (num_tokens > 64) {
        chunk_size = 1024;
    }
    int num_chunks = hidden_size / chunk_size;
    const int num_threads = 128;

    dim3 grid(num_tokens, num_chunks, 1);
    dim3 block(num_threads, 1, 1);

    permute_tokens_kernel<T><<<grid, block>>>(dest, src, mappings, hidden_size, chunk_size);
}

torch::Tensor permute_tokens_cuda(torch::Tensor tokens, torch::Tensor mappings) {
    assert(tokens.dim() == 2);
    assert(mappings.dim() == 1);

    int num_tokens = tokens.size(0);
    int hidden_size = tokens.size(1);

    torch::Tensor out = torch::empty_like(tokens);
   
    AT_DISPATCH_REDUCED_FLOATING_TYPES(tokens.scalar_type(), "permute_tokens_cuda", [&] {
        _permute_tokens_cuda<scalar_t>(out.data_ptr<scalar_t>(), tokens.data_ptr<scalar_t>(), mappings.data_ptr<long>(), num_tokens, hidden_size);
    });

    return out;
}