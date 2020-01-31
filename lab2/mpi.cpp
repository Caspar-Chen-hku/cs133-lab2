// Header inclusions, if any...

#include <mpi.h>
#include <stdlib.h>
#include <iostream>



#include "../lab1/gemm.h"
using std::clog;
using std::endl;

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

  int numproc;
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  int aCount = kI*kK/numproc;
  int bCount = kK*kJ;
  int cCount = kI*kJ/numproc;


  clog << "numproc: " << numproc << endl;
  //MPI_Gather(c, cCount, MPI_FLOAT, c_buffer, cCount, MPI_FLOAT, kRoot, MPI_COMM_WORLD);
  float *a_buffer = (float*) std::aligned_alloc(32, aCount);
  //float *b_buffer = (float*) std::aligned_alloc(32, bCount);
  float *c_buffer = (float*) std::aligned_alloc(32, cCount);

  MPI_Scatter(a, aCount*numproc, MPI_FLOAT, a_buffer, aCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  clog << "scattered\n"
  MPI_Bcast( (void*) b, bCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  clog << "broadcasted\n";

  int BLOCK_SIZE_I = kI/8;
  int BLOCK_SIZE_J = kJ/4;
  int BLOCK_SIZE_K = kK/64;
  int index_a, index_b, index_c;

  /*
  int BLOCK_SIZE_I = 64;
  int BLOCK_SIZE_J = 1024;
  int BLOCK_SIZE_K = 8;
  */
  for (int i = 0; i < cCount; ++i) {
    c_buffer[i] = 0;
  }

    for (int i=0; i< kI; i+=BLOCK_SIZE_I){
        for (int k=0; k< kK; k+=BLOCK_SIZE_K){
          for (int j=0; j< kJ; j+=BLOCK_SIZE_J){
            for (int i0=i; i0<i+BLOCK_SIZE_I; i0++){
              index_a = i0*kK+k;
              for (int k0=k; k0<k+BLOCK_SIZE_K; k0++){
                //index_b = k0*kJ+j;
                index_c = i0*kJ+j;
                for (int j0=j; j0<j+BLOCK_SIZE_J; j0++){
                  c_buffer[index_c] += a_buffer[index_a] * b[k0][j0];
                }
                //index_b++; 
                index_c++;
              }
              index_a++;
            }
          }
        }
  }

  clog << "calculated\n";

  MPI_Gather(c_buffer, cCount, MPI_FLOAT, c, cCount*numproc, MPI_FLOAT, 0, MPI_COMM_WORLD);

  clog << "gathered\n";

}
