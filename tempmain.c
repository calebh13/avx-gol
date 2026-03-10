#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "funcs.h"

int rank = 0;
int p = 1;
FILE* logfile;


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int n = 65536;
    int rows = 65536;
    int generations = 10;

    char logname[64];
    sprintf(logname, "log_p%d.txt", rank);
    logfile = fopen(logname, "w");


    if (argc >= 3) {
        n = atoi(argv[1]);
        rows = atoi(argv[2]);
    }

    if (argc >= 4) {
        generations = atoi(argv[3]);
    }

    if (rank == 0) {
        printf("Grid size: %d x %d\n", rows, n);
        printf("Generations: %d\n\n", generations);
    }

    Grid* grid = init_grid(n, rows);

    GenerateInitialGoL(grid);

    if (rank == 0) {
        printf("Initial Grid:\n");
        //print_grid(grid, NULL);
    }

    simulate(grid, generations);

    MPI_Finalize();
    return 0;
}

