// Header inclusions, if any...

#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>



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

  int numproc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int aCount = kI*kK/numproc;
  int bCount = kK*kJ;
  int cCount = kI*kJ/numproc;

  //MPI_Gather(c, cCount, MPI_FLOAT, c_buffer, cCount, MPI_FLOAT, kRoot, MPI_COMM_WORLD);
  float *a_buffer;
  float *b_buffer;
  float *c_buffer;

  if (rank != 0){
    a_buffer = (float*) std::aligned_alloc(32, aCount*sizeof *a_buffer);
    b_buffer = (float*) std::aligned_alloc(32, bCount*sizeof *b_buffer);
    c_buffer = (float*) std::aligned_alloc(32, cCount*sizeof *c_buffer);
  }

  clog << "allocated\n";

  //MPI_Scatter(a, aCount, MPI_FLOAT, a_buffer, aCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  //clog << "scattered\n";
  int rows = kI/numproc;
  int offset = rows;
  MPI_Status status;

  if (rank == 0){
    for (int i=1; i<numproc; i++){
      MPI_Send(&a[offset][0], aCount, MPI_FLOAT, i, 1,
                   MPI_COMM_WORLD);
      offset += rows;
    }
  }else{
    MPI_Recv(a_buffer, aCount, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
  }

  if (rank == 0){
    for (int i=1; i<numproc; i++){
      MPI_Send(b, bCount, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
    }
  }else {
    MPI_Recv(b_buffer, bCount, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);
  }



  //MPI_Bcast( reinterpret_cast<void*>(b), bCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
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

 if (rank != 0){
   std:memset(c_buffer, 0, sizeof(float) * cCount);
 }


    for (int i=0; i< kI/4; i+=BLOCK_SIZE_I){
        for (int k=0; k< kK; k+=BLOCK_SIZE_K){
          for (int j=0; j< kJ; j+=BLOCK_SIZE_J){
            for (int i0=i; i0<i+BLOCK_SIZE_I; i0++){
              if (rank == 0){
                std::memset(c[i0], 0, sizeof(float) * kJ);
              }
              for (int k0=k; k0<k+BLOCK_SIZE_K; k0++){
                for (int j0=j; j0<j+BLOCK_SIZE_J; j0++){
                  if (rank==0){
                    c[i0][j0] += a[i0][k0] * b[k0][j0];
                  }else{
                    c_buffer[i0*kJ+j0] += a_buffer[i0*kK+k0] * b_buffer[k0*kJ+j0];
                  }
                }
              }
            }
          }
        }
  }

/*
index_a = 0;
index_b = 0;
index_c = 0;

for (int i=0; i< kI/4; i++){
  if (rank==0){
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
      index_b = i*kJ;
        for (int k=0; k< kK; k++){
          index_c = i*kJ;
          for (int j=0; j< kJ; j++)
          {
              if (rank==0){
                c[i][j] += a[i][k] * b[k][j];
              }else{
                c_buffer[index_c] += a_buffer[index_a]*b_buffer[index_b];
              }
              index_c++;    
              index_b++;      
          }
          index_a++;
        }
        index_a += kK;
    }
*/


  clog << "calculated\n";

  if (rank != 0){
    MPI_Send(c_buffer, cCount, MPI_FLOAT, 0, 1,
                   MPI_COMM_WORLD);
  }
  if (rank == 0){
    offset = rows;
    for (int i=1; i<numproc; i++){
      MPI_Recv(&c[offset][0], cCount, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
      offset += rows;
    }
  }

  //MPI_Gather(c_buffer, cCount, MPI_FLOAT, c, cCount, MPI_FLOAT, 0, MPI_COMM_WORLD);

  clog << "gathered\n";

}
