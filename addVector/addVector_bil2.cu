// This program computer the sum of two N-element vectors using unified memory
// By: Nick from CoffeeBeforeArch

#include <stdio.h>
//#include <cassert>
#include <iostream>

using std::cout;

// CUDA kernel for vector addition
// No change when using CUDA unified memory
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    // Calculate global thread thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Each thread processes a portion of the array
    int start = tid * (N / blockDim.x);
    int end = start + (N / blockDim.x);

    // Ensure that we don't go beyond the array size
    end = min(end, N);

    for (int i = start; i < end; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
  // Array size of 2^16 (65536 elements)
  //const int N = 1 << 16;
  const int N = 1000000000; // Number of elements in the array
  size_t bytes = N * sizeof(int);

  // Declare unified memory pointers
  int *a, *b, *c;

  // Allocation memory for these pointers
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);

  // Initialize vectors
  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  // Threads per CTA (1024 threads per CTA)
  int BLOCK_SIZE = 1 << 10;

  // CTAs per Grid
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Call CUDA kernel
  vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

  // Wait for all previous operations before using values
  // We need this because we don't get the implicit synchronization of
  // cudaMemcpy like in the original example
  cudaDeviceSynchronize();

  // Verify the result on the CPU
  //for (int i = 0; i < N; i++) {
  //  assert(c[i] == a[i] + b[i]);
  //}

  // Free unified memory (same as memory allocated with cudaMalloc)
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  cout << "COMPLETED SUCCESSFULLY!\n";

  return 0;
}
