// Header inclusions, if any...

#include <mpi.h>
#include <cstring>

#include "../lab1/gemm.h"

// Using declarations, if any...

/*
00 01   00 01    00 01
10 11   10 11    10 11

      part1     part2
proc1: 00 00    01 10
proc2: 01 11    00 01
proc3: 11 10    10 00
proc4: 10 01    11 11

aligned_alloc

get # processors
use lab1 code
scatter/gather from node 0
allcate using aligned_alloc

*/

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...
  int BLOCK_SIZE_I = kI/8;
  int BLOCK_SIZE_J = kJ/4;
  int BLOCK_SIZE_K = kK/64;

  /*
  int BLOCK_SIZE_I = 64;
  int BLOCK_SIZE_J = 1024;
  int BLOCK_SIZE_K = 8;
  */
  #pragma omp parallel for schedule(static) num_threads(8)
  for (int i = 0; i < kI; ++i) {
    //c[i] = (float*) aligned_alloc(32, kJ);
    std::memset(c[i], 0, sizeof(float) * kJ);
  }

    #pragma omp parallel for schedule(static) num_threads(8)
    for (int i=0; i< kI; i+=BLOCK_SIZE_I){
        for (int k=0; k< kK; k+=BLOCK_SIZE_K){
          for (int j=0; j< kJ; j+=BLOCK_SIZE_J){
            for (int i0=i; i0<i+BLOCK_SIZE_I; i0++){
              for (int k0=k; k0<k+BLOCK_SIZE_K; k0++){
                for (int j0=j; j0<j+BLOCK_SIZE_J; j0++){
                  c[i0][j0] += a[i0][k0] * b[k0][j0];
                }
              }
            }
          }
        }
  }
}
