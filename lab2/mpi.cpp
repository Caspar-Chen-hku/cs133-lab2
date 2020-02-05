// Header inclusions, if any...

#include <mpi.h>
#include <stdlib.h>
#include <cstring>
//#include <iostream>

#include "../lab1/gemm.h"
//using std::clog;

// Using declarations, if any...

/*
      part1     part2
proc1: 00 00    01 10
proc2: 01 11    00 01
proc3: 11 10    10 00
proc4: 10 01    11 11
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

  float *a_buffer;
  float *b_buffer;
  float *c_buffer;
  
  a_buffer = (float*) std::aligned_alloc(64, aCount*sizeof *a_buffer);
  b_buffer = (float*) std::aligned_alloc(64, bCount*sizeof *b_buffer);
  c_buffer = (float*) std::aligned_alloc(64, cCount*sizeof *c_buffer);
  std::memset(c_buffer, 0, sizeof(float) * cCount);
/*
  int rows = kI/numproc;
  int offset = rows;
*/
  MPI_Status status;

  /**************SEND BLOCKS OF DATA*******************/
/*
  if (rank == 0){
  memcpy(b_buffer, b, sizeof(float)*bCount);
    for (int i=1; i<numproc; i++){
      MPI_Send(b, bCount, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
    }
  }else{
    MPI_Recv(b_buffer, bCount, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);
  }
*/
if (rank == 0){
  memcpy(b_buffer, b, sizeof(float)*bCount);
}

MPI_Scatter(a, aCount, MPI_FLOAT, a_buffer,
    aCount, MPI_FLOAT, 0,  MPI_COMM_WORLD);
MPI_Bcast(b_buffer, bCount, MPI_FLOAT,
  0, MPI_COMM_WORLD);

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
/***********************CALCULATE*************************/

  int BLOCK_SIZE_I = 16;
  int BLOCK_SIZE_K = 8;
  int BLOCK_SIZE_J = 16;
  float temp;

    for (int i=0; i< kI/numproc; i+=BLOCK_SIZE_I){
      for (int k=0; k< kK; k+=BLOCK_SIZE_K){
        for (int j=0; j< kJ; j+=BLOCK_SIZE_J){
      for (int i0=i; i0<i+BLOCK_SIZE_I; i0++){
        for (int j0=j; j0<j+BLOCK_SIZE_J; j0++){
          temp = 0;
          for (int k0=k; k0<k+BLOCK_SIZE_K; k0++){
              temp += a_buffer[i0*kJ+k0] * b_buffer[k0*kJ+j0];
          }
          c_buffer[i0*kJ+j0] += temp;
          }
        }
      }
      }
  }

/********************GATHER DATA*****************************/
/*
if (rank != 0){
  MPI_Send(c_buffer, cCount, MPI_FLOAT, 0, 1,
                   MPI_COMM_WORLD);
}else{
  offset = rows;
  memcpy(c, c_buffer, sizeof(float) * cCount);
  for (int i=1; i<numproc; i++){
    MPI_Recv(&c[offset][0], cCount, MPI_FLOAT, i, 1,
                   MPI_COMM_WORLD, &status);
      offset += rows;
  }
}
*/

//clog << "gathered\n";

MPI_Gather(c_buffer, cCount, MPI_FLOAT, c, cCount, MPI_FLOAT,
  0, MPI_COMM_WORLD);


/*
if (rank != 0){
  MPI_Isend(c_buffer, cCount, MPI_FLOAT, 0, 1,
                   MPI_COMM_WORLD, &request);
}else{
  offset = rows;
  memcpy(c, c_buffer, sizeof(float) * cCount);
  for (int i=1; i<numproc; i++){
    MPI_Irecv(&c[offset][0], cCount, MPI_FLOAT, i, 1,
                   MPI_COMM_WORLD, &request);
      offset += rows;
      MPI_Wait(&request, &status);
  }
}
*/

}
