// Header inclusions, if any...
#include <cstring>
#include <stdlib.h>

#include "gemm.h"

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...
  //int BLOCK_SIZE_I = kI/8;
  //int BLOCK_SIZE_J = kJ/4;
  //int BLOCK_SIZE_K = kK/64;

  int BLOCK_SIZE_I = 64;
  int BLOCK_SIZE_J = 1024;
  int BLOCK_SIZE_K = 8;

  float a_buffer[64][8];

  #pragma omp parallel for schedule(static) num_threads(8)
    for (int i=0; i< kI; i+=BLOCK_SIZE_I){
        for (int k=0; k< kK; k+=BLOCK_SIZE_K){
          for (int ii=i; ii<i+64; ii++){
            for (int kk=k; kk<k+8; kk++){
              a_buffer[ii-i][kk-k] = a[ii][kk];
            }
          }
          for (int j=0; j< kJ; j+=BLOCK_SIZE_J){
            for (int i0=i; i0<i+BLOCK_SIZE_I; i0++){
              std::memset(c[i0], 0, sizeof(float) * kJ);
              for (int j0=j; j0<j+BLOCK_SIZE_J; j0++){
                float temp = c[i0][j0];
                for (int k0=k; k0<k+BLOCK_SIZE_K; k0++){
                  //c[i0][j0] += a[i0][k0] * b[k0][j0];
                  temp += a_buffer[i0-i][k0-k] * b[k0][j0];
                }
                c[i0][j0] = temp;
              }
            }
          }
        }
  }

}
