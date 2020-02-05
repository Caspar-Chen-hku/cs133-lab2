// Header inclusions, if any...

#include <mpi.h>
#include <stdlib.h>
#include <cstring>
//#include <iostream>

#include "../lab1/gemm.h"
//using std::clog;

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

  /*************** INITIALIZE *****************/
  int numproc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int aCount = kI*kK/numproc;
  int bCount = kK*kJ;
  int cCount = kI*kJ/numproc;

  /*
  int half_size = kI/2;
  int count = half_size*half_size;
  */

  
  float *a_buffer;
  float *b_buffer;
  float *c_buffer;
  
  if (rank != 0){
    a_buffer = (float*) std::aligned_alloc(64, aCount*sizeof *a_buffer);
    b_buffer = (float*) std::aligned_alloc(64, bCount*sizeof *b_buffer);
    c_buffer = (float*) std::aligned_alloc(64, cCount*sizeof *c_buffer);
  }

  int rows = kI/numproc;
  int offset = rows;

  MPI_Status status;

  /**************SEND BLOCKS OF DATA*******************/


  if (rank == 0){
    for (int i=1; i<numproc; i++){
      MPI_Send(&a[offset][0], aCount, MPI_FLOAT, i, 1,
                   MPI_COMM_WORLD);
      MPI_Send(b, bCount, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
      offset += rows;
    }
  }else{
    MPI_Recv(a_buffer, aCount, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(b_buffer, bCount, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);
  }


MPI_Scatter(a, aCount, MPI_FLOAT, a_buffer,
    aCount, MPI_FLOAT, 0,  MPI_COMM_WORLD);

/*
MPI_Request request;
  if (rank == 0){
    for (int i=1; i<numproc; i++){
      MPI_Isend(&a[offset][0], aCount, MPI_FLOAT, i, 1,
                   MPI_COMM_WORLD, &request);
      offset += rows;
    }
  }else{
    MPI_Irecv(a_buffer, aCount, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, &status);
  }

  if (rank == 0){
    for (int i=1; i<numproc; i++){
      MPI_Isend(b, bCount, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &request);
      offset += rows;
    }
  }else{
    MPI_Irecv(b_buffer, bCount, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, &status);
  }
*/
 /*
 if (rank == 0) {
   for (int i=0; i<half_size; i++){
     MPI_Send(&a[i][half_size], half_size, MPI_FLOAT, 1, i,
                   MPI_COMM_WORLD);
   }
   for (int i=half_size; i<kI; i++){
     MPI_Send(&a[i][0], half_size, MPI_FLOAT, 3, i-half_size,
                   MPI_COMM_WORLD);
    MPI_Send(&a[i][half_size], half_size, MPI_FLOAT, 2, i-half_size,
                   MPI_COMM_WORLD);
   }
 }else{
   for (int i=0; i<half_size; i++){
     MPI_Recv(&a_buffer[i*half_size], half_size, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
   }
 }

  if (rank == 0) {
   for (int i=0; i<half_size; i++){
     MPI_Send(&b[i][half_size], half_size, MPI_FLOAT, 3, i,
                   MPI_COMM_WORLD);
   }
   for (int i=half_size; i<kI; i++){
     MPI_Send(&b[i][0], half_size, MPI_FLOAT, 2, i-half_size,
                   MPI_COMM_WORLD);
    MPI_Send(&b[i][half_size], half_size, MPI_FLOAT, 1, i-half_size,
                   MPI_COMM_WORLD);
   }
 }else{
   for (int i=0; i<half_size; i++){
     MPI_Recv(&b_buffer[i*half_size], half_size, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
   }
 }
*/
/***********************CALCULATE*************************/


  int BLOCK_SIZE_I = 256;
  int BLOCK_SIZE_K = 32;
  int BLOCK_SIZE_J = kJ/2;
  int index_a, index_b, index_c;

    for (int i=0; i< kI/numproc; i+=BLOCK_SIZE_I){
      for (int k=0; k< kK; k+=BLOCK_SIZE_K){
        for (int j=0; j< kJ; j+=BLOCK_SIZE_J){
      for (int i0=i; i0<i+BLOCK_SIZE_I; i0++){
        index_a = i0*kJ+k;
        for (int k0=k; k0<k+BLOCK_SIZE_K; k0++){
          index_b = k0*kJ+j;
          index_c = i0*kJ+j;
          for (int j0=j; j0<j+BLOCK_SIZE_J; j0++){
            
                  if (rank==0){
                    c[i0][j0] += a[i0][k0] * b[k0][j0];
                  }else{
                    c_buffer[index_c] += a_buffer[index_a] * b_buffer[index_b];
                    index_b++;
                    index_c++;
                  }
            }
            index_a++;
          }
        }
      }
      }
  }

/*
 if (rank == 0) {
   for (int i=0; i<half_size; i++){
     MPI_Send(&a[i][0], half_size, MPI_FLOAT, 1, i,
                   MPI_COMM_WORLD);
   }
   for (int i=half_size; i<kI; i++){
     MPI_Send(&a[i][0], half_size, MPI_FLOAT, 2, i-half_size,
                   MPI_COMM_WORLD);
    MPI_Send(&a[i][half_size], half_size, MPI_FLOAT, 3, i-half_size,
                   MPI_COMM_WORLD);
   }
 }else{
   for (int i=0; i<half_size; i++){
     MPI_Recv(&a_buffer[i*half_size], half_size, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
   }
 }

  if (rank == 0) {
   for (int i=0; i<half_size; i++){
     MPI_Send(&b[i][0], half_size, MPI_FLOAT, 2, i,
                   MPI_COMM_WORLD);
    MPI_Send(&b[i][half_size], half_size, MPI_FLOAT, 1, i,
                   MPI_COMM_WORLD);
   }
   for (int i=half_size; i<kI; i++){
    MPI_Send(&b[i][half_size], half_size, MPI_FLOAT, 3, i-half_size,
                   MPI_COMM_WORLD);
   }
 }else{
   for (int i=0; i<half_size; i++){
     MPI_Recv(&b_buffer[i*half_size], half_size, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
   }
 }

if (rank == 0){
     for (int i=0; i< kI/2; i++){
      for (int k=0; k< kK/2; k++){
        for (int j=0; j< kJ/2; j++)
        {
          c[i][j] += a[i][k+half_size] * b[k+half_size][j];
        }
      }
    }
 }else{
   multiply(a_buffer, b_buffer, c_buffer);
 }

*/

  /*
  if (rank != 0){
    for (int i=0; i<half_size; i++){
     MPI_Send(&c_buffer[i*half_size], half_size, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
   }
  }else{
    for (int i=0; i<half_size; i++){
      MPI_Recv(&c[i][half_size], half_size, MPI_FLOAT, 1, i, MPI_COMM_WORLD, &status);
    }
    for (int i=half_size; i<kI; i++){
      MPI_Recv(&c[i][0], half_size, MPI_FLOAT, 2, i-half_size, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[i][half_size], half_size, MPI_FLOAT, 3, i-half_size, MPI_COMM_WORLD, &status);
    }
  }
*/

/********************GATHER DATA*****************************/


if (rank != 0){
  MPI_Send(c_buffer, cCount, MPI_FLOAT, 0, 1,
                   MPI_COMM_WORLD);
}else{
  offset = rows;
  for (int i=1; i<numproc; i++){
    MPI_Recv(&c[offset][0], cCount, MPI_FLOAT, i, 1,
                   MPI_COMM_WORLD, &status);
      offset += rows;
  }
}


/*
MPI_Gather(c_buffer, cCount, MPI_FLOAT, c, cCount, MPI_FLOAT,
  0, MPI_COMM_WORLD);
*/

/*
if (rank != 0){
  MPI_Isend(c_buffer, cCount, MPI_FLOAT, 0, 1,
                   MPI_COMM_WORLD, &request);
}else{
  offset = rows;
  for (int i=1; i<numproc; i++){
    MPI_Irecv(&c[offset][0], cCount, MPI_FLOAT, i, 1,
                   MPI_COMM_WORLD, &request);
      offset += rows;
      MPI_Wait(&request, &status);
  }
}
*/

}
