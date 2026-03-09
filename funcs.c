#include "funcs.h"
#include <mpi.h>

struct Grid {
    char** grid;
    size_t n;
    size_t rows;
};

Grid* init_grid(int n, int rows)
{
    LOG("Entered init_grid");
    Grid* grid = malloc(sizeof(Grid));
    if (!grid) return NULL;
    LOG("malloc'd grid");
    grid->grid = malloc(rows * sizeof(char*));
    if (!grid->grid) return NULL;
    LOG("malloc'd grid->grid");
    for (int i = 0; i < rows; i++) {
        grid->grid[i] = malloc(n * sizeof(char));
        if (!grid->grid[i]) return NULL;
        // LOG("malloc'd grid->grid[%d]", i);
    }

    grid->n = n;
    grid->rows = rows;
    LOG("successfully initialized grid");
    return grid;
}

void scatter_seeds()
{
    unsigned int seed;

    if (rank == 0) {
        LOG("Allocating seed array");
        unsigned int *seeds = malloc(p * sizeof(unsigned int));

        LOG("Generating seeds");
        srand(12345);

        for (int i = 0; i < p; i++) {
            seeds[i] = rand() % BIGPRIME + 1;
            LOG("Seed[%d]=%u", i, seeds[i]);
        }

        LOG("MPI_Scatter seeds");
        MPI_Scatter(seeds, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0,
                MPI_COMM_WORLD);

        free(seeds);
        LOG("Freed seed array");
    } else {
        LOG("Waiting for MPI_Scatter seed");
        MPI_Scatter(NULL, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0,
                MPI_COMM_WORLD);
    }

    LOG("Received seed %u", seed);

    srand(seed);
}

void GenerateInitialGoL(Grid* local_grid) 
{
    LOG("Entered GenerateInitialGoL (n=%zu rows=%zu)", local_grid->n, local_grid->rows);

    scatter_seeds();

    LOG("Generating local grid");
    for(int i = 0; i < local_grid->rows; i++) {
        for(int j = 0; j < local_grid->n; j++){
            local_grid->grid[i][j] = rand() % 2 == 0;
        }
    }

    LOG("Exiting GenerateInitialGoL");
}

void free_grid(Grid* local_grid) {
    LOG("Entered free_grid rows=%zu", local_grid->rows);

    for(int i = 0; i < local_grid->rows; i++){
        LOG("Freeing row %d", i);
        free(local_grid->grid[i]);
    }

    free(local_grid->grid);
    free(local_grid);

    LOG("Exiting free_grid");
}

// User must call MPI_Wait on send_req and recv_req before attempting to use the relevant buffers
// This is not critical for the send buffer (so long as it isn't freed), but is extremely important for the recv buffer
static void sendLowerRecvUpper(Grid* local_grid, char* upper_row, MPI_Request* send_req, MPI_Request* recv_req)
{
    LOG("Entered sendLowerRecvUpper");

    MPI_Request req;

    int send_target = (rank + 1) % p;
    int recv_source = (rank + p - 1) % p;

    LOG("MPI_Isend lower row -> p%d", send_target);

    MPI_Isend(
        local_grid->grid[local_grid->rows - 1],
        local_grid->n,
        MPI_CHAR,
        send_target,
        0,
        MPI_COMM_WORLD,
        send_req
    );

    LOG("MPI_Recv upper row <- p%d", recv_source);

    MPI_Irecv(
        upper_row,
        local_grid->n,
        MPI_CHAR,
        recv_source,
        MPI_ANY_TAG,
        MPI_COMM_WORLD,
        recv_req
    );

    LOG("Exiting sendLowerRecvUpper");
}

// User must call MPI_Wait on send_req and recv_req before attempting to use the relevant buffers
// This is not critical for the send buffer (so long as it isn't freed), but is extremely important for the recv buffer
static void sendUpperRecvLower(Grid* local_grid, int* lower_row, MPI_Request* send_req, MPI_Request* recv_req)
{
    LOG("Entered sendUpperRecvLower");

    MPI_Request req;

    int send_target = (rank + p - 1) % p;
    int recv_source = (rank + 1) % p;

    LOG("MPI_Isend upper row -> p%d", send_target);

    MPI_Isend(
        local_grid->grid[0],
        local_grid->n,
        MPI_CHAR,
        send_target,
        0,
        MPI_COMM_WORLD,
        send_req
    );

    LOG("MPI_Irecv lower row <- p%d", recv_source);

    MPI_Irecv(
        lower_row,
        local_grid->n,
        MPI_CHAR,
        recv_source,
        MPI_ANY_TAG,
        MPI_COMM_WORLD,
        recv_req
    );

    LOG("Exiting sendUpperRecvLower");
}

/// @brief Simulates a game of life for the generations and size specified
/// @param g number of generations
void simulate(Grid* local_grid, int g){
    assert(g > 0);
    /*
    Step 1: Generate the initial game of life
        -Determine and validate the number of rows going to each process
        -Allocate an array for this process' local_grid to go
        -Allocate an array for this process' output_grid to go
    */

    /*
    Step 2: ISend and Irecieve HALO
        -Isend the information required to the proper process
        -Irecieve the information required to the proper process
        -Save the request references somewhere so that we can wait on them later
    */

    /*
    Step 3: Calculate all of the vectorizable cells for current row
        -Move in 512 bit windows, then 256, then 128, then . . . 
        -Make sure to save the sundogs for a row somewhere for easy calcs later
    */

    /*
    Step 4: Calculate the sun dogs for current row
        -Use the information from the vector calculations to determine the values for sun dogs
    */

    /*
    Step 5: Repeat 3-4 until done with non-HALO rows
    */

    /*
    Step 6: Calculate HALO rows
        -Wait on the earlier requests to ensure info has been recieved
        -Manually create upper middle and lower rows
        -Manually call the determine state stuff
    */

    /*
    Step 7: Output
        -Print to file -- as needed --
        - Swap grid references
    */
}

/*
 * Must not be called on an element on the boundary (far left/right columns, or top/bottom rows)
 * That is, all the arrays must be at least length 34, and should be passed in as arr + 1.
 * i.e, arr[-1] and arr[33] must be valid indices.
*/
__m256i calculate_neighbors256(const char* lower_arr, const char* middle_arr, const char* upper_arr) {
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

    __m256i ge3 = _mm256_cmpgt_epi8(neighbor_counts, two); // v > 2 -> v >= 3
    __m256i le5 = _mm256_cmpgt_epi8(six, neighbor_counts); // 6 > v -> v <= 5

    return _mm256_and_si256(ge3, le5);
}

/*
 * Must not be called on an element on the boundary (far left/right columns, or top/bottom rows)
 * That is, all the arrays must be at least length 18, and should be passed in as arr + 1.
 * i.e, arr[-1] and arr[18] must be valid indices.
*/
__m128i calculate_neighbors128(const char* lower_arr, const char* middle_arr, const char* upper_arr) {
    LOG("Entering determine_state128");

    LOG("Gathering HALO");
    __m128i upper = _mm_loadu_si128((const __m128i*) upper_arr);
    __m128i lower = _mm_loadu_si128((const __m128i*) lower_arr);

    LOG("Gathering Sun Dogs");
    __m128i left = _mm_loadu_si128((const __m128i*) (middle_arr - 1));
    __m128i right = _mm_loadu_si128((const __m128i*) (middle_arr + 1));

    LOG("Gathering NE / NW / SE / SW");
    __m128i upperLeft = _mm_loadu_si128((const __m128i*) (upper_arr - 1));
    __m128i upperRight = _mm_loadu_si128((const __m128i*) (upper_arr + 1));

    __m128i lowerLeft = _mm_loadu_si128((const __m128i*) (lower_arr - 1));
    __m128i lowerRight = _mm_loadu_si128((const __m128i*) (lower_arr + 1));

    __m128i northSouthSum = _mm_add_epi8(upper, lower);
    __m128i eastWestSum = _mm_add_epi8(left, right);
    __m128i northeastNorthwestSum = _mm_add_epi8(upperLeft, upperRight);
    __m128i southeastSouthwestSum = _mm_add_epi8(lowerLeft, lowerRight);

    __m128i cardinalSums = _mm_add_epi8(northSouthSum, eastWestSum);
    __m128i diagonalSums = _mm_add_epi8(northeastNorthwestSum, southeastSouthwestSum);

    LOG("Exiting determine_state128");
    return _mm_add_epi8(cardinalSums, diagonalSums);
}

__m128i determine_state128(__m128i neighbor_counts)
{
    __m128i two = _mm_set1_epi8(2);
    __m128i six = _mm_set1_epi8(6);

    __m128i ge3 = _mm_cmpgt_epi8(neighbor_counts, two); // v > 2  → v >= 3
    __m128i le5 = _mm_cmpgt_epi8(six, neighbor_counts); // 6 > v  → v <= 5

    return _mm_and_si128(ge3, le5);
}