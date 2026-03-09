#include "funcs.h"

#define BIGPRIME 85935431U
#define OUTFILE_FORMAT "p%dout"

int rank, p;
FILE* logfile;

#define LOG(...) \
do { \
    if(logfile){ \
        fprintf(logfile, "p%d: ", rank); \
        fprintf(logfile, __VA_ARGS__); \
        fprintf(logfile, "\n"); \
        fflush(logfile); \
    } \
} while(0)

void GenerateInitialGoL(int n, int rows, char** local_grid) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    char logname[64];
    sprintf(logname, "log_p%d.txt", rank);
    logfile = fopen(logname, "w");

    LOG("Entered GenerateInitialGoL (n=%d rows=%d)", n, rows);

    unsigned int seed;

    if(rank == 0) {
        LOG("Allocating seed array");
        unsigned int *seeds = malloc(p * sizeof(unsigned int));

        LOG("Generating seeds");
        srand(12345);

        for(int i = 0; i < p; i++) {
            seeds[i] = rand() % BIGPRIME + 1;
            LOG("Seed[%d]=%u", i, seeds[i]);
        }

        LOG("MPI_Scatter seeds");
        MPI_Scatter(seeds, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        free(seeds);
        LOG("Freed seed array");
    } 
    else {
        LOG("Waiting for MPI_Scatter seed");
        MPI_Scatter(NULL, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    LOG("Received seed %u", seed);

    srand(seed);

    LOG("Generating local grid");
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < n; j++){
            local_grid[i][j] = rand() % 2 == 0;
        }
    }

    LOG("Exiting GenerateInitialGoL");
}

void free_grid(int n, int rows, char** local_grid) {
    LOG("Entered free_grid rows=%d", rows);

    for(int i = 0; i < rows; i++){
        LOG("Freeing row %d", i);
        free(local_grid[i]);
    }

    free(local_grid);

    LOG("Exiting free_grid");
}

/*
 * Must not be called on an element on the boundary (far left/right columns, or top/bottom rows)
 * That is, all the arrays must be at least length 34, and should be passed in as arr + 1.
 * i.e, arr[-1] and arr[33] must be valid indices.
*/
__m256i calculate_neighbors256(const char* lower_arr, const char* middle_arr, const char* upper_arr) {
    printf("It got called\n");
    LOG("Entering determine_state256");

    LOG("Gathering HALO");
    __m256i upper = _mm256_loadu_si256((const __m256i*) upper_arr);
    __m256i lower = _mm256_loadu_si256((const __m256i*) lower_arr);

    LOG("Gathering Sun Dogs");
    __m256i left = _mm256_loadu_si256((const __m256i*) (middle_arr - 1));
    __m256i right = _mm256_loadu_si256((const __m256i*) (middle_arr + 1));

    LOG("Gathering NE / NW / SE / SW");
    __m256i upperLeft = _mm256_loadu_si256((const __m256i*) (upper_arr - 1));
    __m256i upperRight = _mm256_loadu_si256((const __m256i*) (upper_arr + 1));

    __m256i lowerLeft = _mm256_loadu_si256((const __m256i*) (lower_arr - 1));
    __m256i lowerRight = _mm256_loadu_si256((const __m256i*) (lower_arr + 1));

    __m256i northSouthSum = _mm256_add_epi8(upper, lower);
    __m256i eastWestSum = _mm256_add_epi8(left, right);
    __m256i northeastNorthwestSum = _mm256_add_epi8(upperLeft, upperRight);
    __m256i southeastSouthwestSum = _mm256_add_epi8(lowerLeft, lowerRight);

    __m256i cardinalSums = _mm256_add_epi8(northSouthSum, eastWestSum);
    __m256i diagonalSums = _mm256_add_epi8(northeastNorthwestSum, southeastSouthwestSum);

    LOG("Exiting determine_state256");
    return _mm256_add_epi8(cardinalSums, diagonalSums);
}

__m256i determine_state256(__m256i neighbor_counts)
{
    __m256i two = _mm256_set1_epi8(2);
    __m256i six = _mm256_set1_epi8(6);

    __m256i ge3 = _mm256_cmpgt_epi8(neighbor_counts, two); // v > 2  → v >= 3
    __m256i le5 = _mm256_cmpgt_epi8(six, neighbor_counts); // 6 > v  → v <= 5

    return _mm256_and_si256(ge3, le5);
}