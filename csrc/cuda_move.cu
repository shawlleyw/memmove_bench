#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#include <torch/torch.h>
#include <torch/extension.h>
#include <assert.h>
#include <cstring>
#include <string>

#include "cuda_move.h"

template <class T, int CHUNK_SIZE>
__global__ void permute_tokens_kernel(T *d_out, T *d_in, long *mappings, const int hidden_size) {
    int token_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int num_warps = blockDim.x / 32;

    int tid = threadIdx.x;
    constexpr int WARPSIZE = 32;
    int wid = tid / WARPSIZE;

    int p = mappings[token_id];

    int task_per_warp = CHUNK_SIZE / num_warps;

    int base = chunk_id * CHUNK_SIZE + wid * task_per_warp;

    // TODO: deal with fp16 and bf16
    half2 *d_in_half2 = (half2 *)(d_in + token_id * hidden_size);
    half2 *dest_half2 = (half2 *)(d_out + p * hidden_size);

    task_per_warp /= 2;
    base /= 2;

    #pragma unroll
    for (int i = tid; i < task_per_warp; i += WARPSIZE) {
        dest_half2[base + i] = d_in_half2[base + i];
    }
}

template <class T>
void _permute_tokens_cuda(T *dest, T *src, long *mappings, int num_tokens, int hidden_size) {

    constexpr int chunk_size = 1024;
    // if (num_tokens > 64) {
    //     chunk_size = 1024;
    // }
    int num_chunks = hidden_size / chunk_size;
    const int num_threads = 128;

    dim3 grid(num_tokens, num_chunks, 1);
    dim3 block(num_threads, 1, 1);

    permute_tokens_kernel<T, chunk_size><<<grid, block>>>(dest, src, mappings, hidden_size);
}

torch::Tensor permute_tokens_cuda(torch::Tensor tokens, torch::Tensor mappings) {
    assert(tokens.dim() == 2);
    assert(mappings.dim() == 1);
    assert(tokens.size(0) == mappings.size(0));

    int num_tokens = tokens.size(0);
    int hidden_size = tokens.size(1);

    torch::Tensor out = torch::empty_like(tokens);
   
    AT_DISPATCH_REDUCED_FLOATING_TYPES(tokens.scalar_type(), "permute_tokens_cuda", [&] {
        _permute_tokens_cuda<scalar_t>(out.data_ptr<scalar_t>(), tokens.data_ptr<scalar_t>(), mappings.data_ptr<long>(), num_tokens, hidden_size);
    });

    return out;
}