#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include "funcs.h"

int main(int argc, char* argv[]) {
    printf("Hello World\n");
    
    MPI_Init(&argc, &argv);

    const int n = 16;
    const int G = 2;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    assert(n > p && n % p == 0);

    MPI_Finalize();
    return 0;
}