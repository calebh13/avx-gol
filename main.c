#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include "funcs.h"

int rank, p;
FILE* logfile;

int main(int argc, char* argv[]) {
    printf("Hello World\n");
    
    MPI_Init(&argc, &argv);

    const int n = 16;
    const int g = 2;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    char logname[64];
    sprintf(logname, "log_p%d.txt", rank);
    logfile = fopen(logname, "w");

    assert(n > p && n % p == 0);
    
    Grid* local_grid = init_grid(n, n / p);
    GenerateInitialGoL(local_grid);
    simulate(local_grid, g);

    free_grid(local_grid);
    LOG("Closing logfile");
    fclose(logfile);

    MPI_Finalize();
    return 0;
}