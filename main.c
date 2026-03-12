#include "gol.h"

#define SEC_TO_US 1000000

int rank;
int p;
FILE* logfile;

extern double comm_time;


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int n = 32768 * 2;
    int rows = n / p;
    int generations = 40;

    char logname[64];
    sprintf(logname, LOGFILE_FORMAT, rank);
    logfile = fopen(logname, "w");

    if (argc >= 2) {
        n = atoi(argv[1]);
        rows = n / p;
    }

    if (argc >= 3) {
        generations = atoi(argv[2]);
    }

    assert(n > p && n % p == 0);

    Grid* grid = init_grid(n, rows);

    GenerateInitialGoL(grid);

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    simulate(grid, generations, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    double max_comm_time;

    MPI_Reduce(
        &comm_time,
        &max_comm_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) { 
        printf("FORMAT:p,n,total runtime,avg time/gen,total comm time,total comp time\n");

        double total_runtime_us = (end - start) * SEC_TO_US;
        double avg_gen_time_us = total_runtime_us / generations;
        double total_comm_time_us = max_comm_time * SEC_TO_US;
        double comp_time_us = total_runtime_us - total_comm_time_us;

        printf("%d,%d,%.0f,%.0f,%.0f,%.0f\n",
            p,
            n,
            total_runtime_us,
            avg_gen_time_us,
            total_comm_time_us,
            comp_time_us);
    }

    fclose(logfile);

    MPI_Finalize();
    return 0;
}