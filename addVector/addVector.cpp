#include <iostream>
#include <cassert>
#include <cstdlib> // For rand()

using std::cout;

// CPU function for vector addition
void vectorAdd(int *a, int *b, int *c, int N) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  // Array size of 2^16 (65536 elements)
  const int N = 1 << 16;
  //size_t bytes = N * sizeof(int);

  // Declare and allocate memory for host vectors
  int *a = new int[N];
  int *b = new int[N];
  int *c = new int[N];

  // Initialize vectors
  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  // Call CPU vector addition function
  vectorAdd(a, b, c, N);

  // Verify the result on the CPU
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }

  // Free host memory
  delete[] a;
  delete[] b;
  delete[] c;

  cout << "COMPLETED SUCCESSFULLY!\n";

  return 0;
}
