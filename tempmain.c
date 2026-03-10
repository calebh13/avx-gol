#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "funcs.h"

int rank;
int p;
FILE* logfile;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int n = 1024;
    int rows = n / p;
    int generations = 10;

    char logname[64];
    sprintf(logname, "log_p%d.txt", rank);
    logfile = fopen(logname, "w");


    if (argc >= 2) {
        n = atoi(argv[1]);
        rows = n / p;
    }

    if (argc >= 3) {
        generations = atoi(argv[2]);
    }

    if (rank == 0) {
        printf("Grid size: %d x %d\n", rows, n);
        printf("Generations: %d\n\n", generations);
    }

    Grid* grid = init_grid(n, rows);

    GenerateInitialGoL(grid);

    if (rank == 0) {
        printf("Initial Grid:\n");
        // print_grid(grid, NULL);
    }

    simulate(grid, generations);
    fclose(logfile);

    MPI_Finalize();
    return 0;
}

