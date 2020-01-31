#include <iostream>

#include <mpi.h>
#include <stdlib.h>

#include "../lab1/gemm.h"

using std::clog;
using std::endl;

int main(int argc, char** argv) {
  int rank, numproc;
  const int kRoot = 0;
  float (*a)[kK] = nullptr;
  float (*b)[kJ] = nullptr;
  float (*c)[kJ] = nullptr;
  float (*c_base)[kJ] = nullptr;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //clog << "\nrank: " << rank << ", numproc: " << numproc << "\n";

  if (rank == kRoot) {
    a = new float[kI][kK];
    b = new float[kK][kJ];
    c = new float[kI][kJ];
    c_base = new float[kI][kJ];

    Init(a, b);

    GemmBaseline(a, b, c_base);

    clog << "\nRun parallel GEMM with MPI\n";
  }

    //float **a_buffer = new float*[kI/4];
    //float **b_buffer = new float*[kK];
    //float **c_buffer = new float*[kI/4];

    float (*a_buffer)[kK] = nullptr;
    float (*b_buffer)[kJ] = nullptr;
    float (*c_buffer)[kJ] = nullptr;

    for (int i=0; i<kI/4; i++){
      a_buffer[i] = (*float (*)[kK]) std::aligned_alloc(32, kK);
    }

    for (int i=0; i<kK; i++){
      b_buffer[i] = (*float (*)[kJ]) std::aligned_alloc(32, kJ);
    }

    for (int i=0; i<kI/4; i++){
      c_buffer[i] = (*float (*)[kJ]) std::aligned_alloc(32, kJ);
    }

    int aCount = kI*kK/numproc;
    int bCount = kK*kJ;
    MPI_Scatter(a, aCount, MPI_FLOAT, a_buffer, aCount, MPI_FLOAT, kRoot, MPI_COMM_WORLD);
    MPI_Scatter(b, bCount, MPI_FLOAT, b_buffer, bCount, MPI_FLOAT, kRoot, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  double begin = MPI_Wtime();
  GemmParallelBlocked(a_buffer, b_buffer, c_buffer);
  MPI_Barrier(MPI_COMM_WORLD);
  double end = MPI_Wtime();

  int cCount = kI*kJ/numproc;
  MPI_Gather(c, cCount, MPI_FLOAT, c_buffer, cCount, MPI_FLOAT, kRoot, MPI_COMM_WORLD);
  if (rank == kRoot) {

    double run_time = end - begin;
    float gflops = 2.0 * kI * kJ * kK / (run_time * 1e9);
    clog << "Time: " << run_time << " s\n";
    clog << "Perf: " << gflops << " GFlops\n";

    bool fail = false;
    if (Diff(c_base, c_buffer) != 0) {
      fail = true;
    }

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_base;

    if (fail) {
      return 1;
    }
  }

  MPI_Finalize();
  return 0;
}
