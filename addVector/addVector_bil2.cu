// This program computes the sum of two N-element vectors using unified memory
// By: Nick from CoffeeBeforeArch
// FIXED VERSION

#include <stdio.h>
#include <iostream>
#include <cassert>

using std::cout;

// Error checking macro
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernel for vector addition - CORRECTED
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    // Calculate global thread ID
    // This is the CORRECT way - each thread processes ONE element
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Ensure thread doesn't go out of bounds
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
  // Array size (reduced from 1 billion to 100 million to fit in memory)
  const int N = 100000000; // 100 million elements
  size_t bytes = N * sizeof(int);

  // Declare unified memory pointers
  int *a, *b, *c;

  // Allocate memory for these pointers with error checking
  cudaCheck(cudaMallocManaged(&a, bytes));
  cudaCheck(cudaMallocManaged(&b, bytes));
  cudaCheck(cudaMallocManaged(&c, bytes));

  cout << "Initializing vectors...\n";
  
  // Initialize vectors
  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  cout << "Running kernel...\n";

  // Threads per block (1024 threads per block)
  int BLOCK_SIZE = 1 << 10;  // 1024

  // Blocks per grid
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  printf("Grid: %d blocks, Block: %d threads\n", GRID_SIZE, BLOCK_SIZE);

  // Call CUDA kernel
  vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

  // Check for kernel launch errors
  cudaCheck(cudaGetLastError());

  // Wait for all previous operations before using values
  cudaCheck(cudaDeviceSynchronize());

  cout << "Verifying results...\n";

  // Verify the result on the CPU (sample check)
  bool correct = true;
  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (c[i] != a[i] + b[i]) {
      if (errors < 10) {  // Print first 10 errors
        printf("Error at index %d: c[%d]=%d, expected %d\n", i, i, c[i], a[i] + b[i]);
      }
      errors++;
      correct = false;
    }
  }

  if (correct) {
    cout << "VERIFICATION PASSED!\n";
  } else {
    printf("VERIFICATION FAILED! Found %d errors\n", errors);
  }

  // Free unified memory
  cudaCheck(cudaFree(a));
  cudaCheck(cudaFree(b));
  cudaCheck(cudaFree(c));

  cout << "COMPLETED!\n";

  return 0;
}
