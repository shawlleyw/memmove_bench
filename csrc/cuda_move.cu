#include <cuda_runtime.h>
#include <iostream>

template <class T, class E>
__global__ void permute_tokens_kernel(const T *d_out, const T *d_in, const E *mappings, const int hidden_size, const int chunk_size) {
    int token_id = blockIdx.x;
    int chunk_id = blockIdx.y;

    int tid = threadIdx.x;

    int p = mappings[token_id];
    T *dest = d_out + p * hidden_size * sizeof(T);

    for (int i = 0; i < chunk_size; i += blockDim.x) {
        dest[chunk_id * chunk_size + tid + i] = d_in[chunk_id * chunk_size + tid + i];
    }
}


template <class T, class E>
void permute_tokens_cuda(T *dest, const T *src, const E *mappings, int num_tokens, int hidden_size) {
    const int chunk_size = 256;
    int num_chunks = hidden_size / chunk_size;

    const int num_threads = 128;

    dim3 grid(num_chunks, num_chunks, 1);
    dim3 block(num_threads, 1, 1);

    permute_tokens_kernel<<<grid, block>>>(dest, src, mappings, hidden_size, chunk_size);
}