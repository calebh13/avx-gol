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

    // printf("\033[H\033[2J");

    if (argc >= 2) {
        n = atoi(argv[1]);
        rows = n / p;
    }

    if (argc >= 3) {
        generations = atoi(argv[2]);
    }

    Grid* grid = init_grid(n, rows);

    GenerateInitialGoL(grid);

    simulate(grid, generations);
    fclose(logfile);

    MPI_Finalize();
    return 0;
}

