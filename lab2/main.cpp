#include <iostream>

#include <mpi.h>

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

    int aCount = kI*kK/numproc;
    MPI_Scatter(a, aCount, MPI_FLOAT, a, aCount, MPI_FLOAT, kRoot, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double begin = MPI_Wtime();
  GemmParallelBlocked(a, b, c);
  MPI_Barrier(MPI_COMM_WORLD);
  double end = MPI_Wtime();

  if (rank == kRoot) {
    int cCount = kI*kJ/numproc;
    MPI_Gather(c, cCount, MPI_FLOAT, c, cCount, MPI_FLOAT, kRoot, MPI_COMM_WORLD);

    double run_time = end - begin;
    float gflops = 2.0 * kI * kJ * kK / (run_time * 1e9);
    clog << "Time: " << run_time << " s\n";
    clog << "Perf: " << gflops << " GFlops\n";

    bool fail = false;
    if (Diff(c_base, c) != 0) {
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
