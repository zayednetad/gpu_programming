#include <iostream>

const int N = 1000000000; // Number of elements in the array
const int BLOCK_SIZE = 256; // Number of threads per block

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int elementsPerBlock) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes a portion of the array
    int start = tid * elementsPerBlock;
    int end = start + elementsPerBlock;

    // Ensure that we don't go beyond the array size
    end = min(end, N);

    for (int i = start; i < end; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Allocate Unified Memory for host vectors
    int *h_a, *h_b, *h_c;
    cudaMallocManaged(&h_a, N * sizeof(int));
    cudaMallocManaged(&h_b, N * sizeof(int));
    cudaMallocManaged(&h_c, N * sizeof(int));

    // Initialize host vectors with data

    // Calculate the number of blocks needed
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch the CUDA kernel
    vectorAdd<<<blocks, BLOCK_SIZE>>>(h_a, h_b, h_c, (N + blocks - 1) / blocks);

    // Ensure all CUDA operations are completed
    cudaDeviceSynchronize();

    // Free allocated memory
    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_c);

    return 0;
}
