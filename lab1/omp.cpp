// Header inclusions, if any...
#include <omp.h>
#include <cstring>

#include "gemm.h"

// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {

  #pragma omp parallel for schedule(static) num_threads(8)
    for (int i=0; i< kI; i++){
      std::memset(c[i], 0, sizeof(float) * kJ);
        for (int k=0; k< kK; k++){
          for (int j=0; j< kJ; j++)
          {
              c[i][j] += a[i][k] * b[k][j];
          }
        }
    }
}