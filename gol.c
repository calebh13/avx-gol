#include "gol.h"
#include <mpi.h>

// If true, p0 will read each process's output file in order
// and print the full matrix to stdout.
// Should only be set for debugging.
#define PRINT_FULL_MATRIX 0

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

void print_grid(const Grid* g, FILE* out) {
    if (!out) out = stdout;  // default to stdout if NULL
    LOG("Entered Print_grid");
    for (size_t r = 0; r < g->rows; r++) {
        for (size_t c = 0; c < g->n; c++) {
            fprintf(out, "%d ", g->grid[r][c]);
        }
        fprintf(out, "\n");
    }
    fprintf(out, "\n");
    fflush(out);
    LOG("Exiting Print_Grid");
}

// Reads every process's output file, and prints it to stdout.
// This is REALLY SLOW, and should only be used for testing, since each output file
// has all the generations so far in them.
// For other purposes, use combine.py
void DisplayGoL(int n, int generation)
{
    LOG("Enter print_output_files");

    char filename[64];
    size_t linesize = (n+1) * 2 * sizeof(char);
    char* line = malloc(linesize);

    for(int r = 0; r < p; r++) {
        sprintf(filename, OUTFILE_FORMAT, r);
        LOG("Reading %s", filename);
        FILE* fp = fopen(filename, "r");
        for(int i = 0; i < generation; i++) {
            // We need to skip over all the previous generations
            while (getline(&line, &linesize, fp) != -1) {
                if (line[0] == '\n') break;
            }
        }
        while (getline(&line, &linesize, fp) != -1) {
            if (line[0] == '\n') continue;
            printf("%s", line);
        }
        fflush(stdout);
        fclose(fp);
    }

    free(line);
    printf("\n");
    LOG("Done print_output_files");
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

    LOG("MPI_Irecv upper row <- p%d", recv_source);

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
static void sendUpperRecvLower(Grid* local_grid, char* lower_row, MPI_Request* send_req, MPI_Request* recv_req)
{
    LOG("Entered sendUpperRecvLower");

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

void simulate(Grid* local_grid, int g) {
    LOG("Entering simulate");
    assert(g >= 0);

    LOG("Calculating block sizes");
    int blocks_256 = (local_grid->n - 2) / 32;
    int remainder = (local_grid->n - 2) % 32;
    int blocks_128 = remainder / 16;
    int blocks_1 = remainder % 16;

    LOG("Allocating HALO buffers and MPI requests");
    char* upper_HALO = malloc(sizeof(char) * local_grid->n);
    char* lower_HALO = malloc(sizeof(char) * local_grid->n);
    MPI_Request upper_send, upper_recv, lower_send, lower_recv;

    LOG("Initializing output grid and output file");
    Grid* output_grid = init_grid(local_grid->n, local_grid->rows);

    char outfile_name[64];
    sprintf(outfile_name, OUTFILE_FORMAT, rank);
    FILE* outfile = fopen(outfile_name, "w");
    
    LOG("Printing initial matrix");
    print_grid(local_grid, outfile);
    if (PRINT_FULL_MATRIX) {
        // all processes must be done printing to their files
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            LOG("Printing all grids to stdout");
            DisplayGoL(local_grid->n, 0);
        }
    }

    for(int current_gen = 1; current_gen <= g; current_gen++) {
        MPI_Barrier(MPI_COMM_WORLD);
        LOG("Starting generation %d", current_gen);

        LOG("Posting HALO sends/receives");
        sendLowerRecvUpper(local_grid, upper_HALO, &lower_send, &upper_recv);
        sendUpperRecvLower(local_grid, lower_HALO, &upper_send, &lower_recv);

        LOG("Calculating interior rows");
        for(int row = 1; row < local_grid->rows - 1; row++) {
            LOG("Processing row %d", row);
            char* upper_row = local_grid->grid[row-1] + 1;
            char* middle_row = local_grid->grid[row] + 1;
            char* lower_row = local_grid->grid[row + 1] + 1;
            char* output_row = output_grid->grid[row] + 1;

            output_grid->grid[row][0] = determine_state1_manual_rows(
                local_grid->grid[row - 1],
                local_grid->grid[row],
                local_grid->grid[row + 1],
                local_grid->n,
                0
            );

            output_grid->grid[row][local_grid->n - 1] = determine_state1_manual_rows(
                local_grid->grid[row - 1],
                local_grid->grid[row],
                local_grid->grid[row + 1],
                local_grid->n,
                local_grid->n - 1
            );

            LOG("Processing 256-bit blocks");
            for(int i = 0; i < blocks_256; i++) {
                _mm256_storeu_si256(
                    (__m256i*)output_row, 
                    determine_state256(
                        lower_row,
                        middle_row,
                        upper_row
                    )
                );
                upper_row += 32;
                middle_row += 32;
                lower_row += 32;
                output_row += 32;
            }

            LOG("Processing 128-bit blocks");
            for(int i = 0; i < blocks_128; i++) {
                _mm_storeu_si128(
                    (__m128i*)output_row, 
                    determine_state128(
                        lower_row,
                        middle_row,
                        upper_row
                    )
                );
                upper_row += 16;
                middle_row += 16;
                lower_row += 16;
                output_row += 16;
            }

            LOG("Processing remaining scalar cells");
            for(int i = 0; i < blocks_1; i++) {
                *output_row = determine_state1(
                    lower_row,
                    middle_row,
                    upper_row
                );
                upper_row += 1;
                middle_row += 1;
                lower_row += 1;
                output_row += 1;
            }
        }

        LOG("Waiting on HALO MPI requests");
        MPI_Wait(&upper_send, MPI_STATUS_IGNORE);
        MPI_Wait(&lower_send, MPI_STATUS_IGNORE);
        MPI_Wait(&upper_recv, MPI_STATUS_IGNORE);
        MPI_Wait(&lower_recv, MPI_STATUS_IGNORE);

        // Calculate top HALO row
        calculate_row(
            upper_HALO,        // received from the process above
            local_grid->grid[0], // middle row (first row)
            local_grid->grid[1], // lower row
            output_grid->grid[0],
            local_grid->n,
            blocks_256,
            blocks_128,
            blocks_1
        );

        // Calculate bottom HALO row
        calculate_row(
            local_grid->grid[local_grid->rows-2], // upper row
            local_grid->grid[local_grid->rows-1], // middle row (last row)
            lower_HALO,                           // received from the process below
            output_grid->grid[local_grid->rows-1],
            local_grid->n,
            blocks_256,
            blocks_128,
            blocks_1
        );

        LOG("Swapping grids for next generation");
        Grid* temp = local_grid;
        local_grid = output_grid;
        output_grid = temp;

        print_grid(local_grid, outfile);
        if (PRINT_FULL_MATRIX) {
            LOG("PRINT_FULL_MATRIX is true");
            // all processes must be done printing to their files
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                LOG("Printing all grids to stdout");
                DisplayGoL(local_grid->n, current_gen);
            }
        }

        LOG("Finished generation %d", current_gen);
    }

    LOG("Cleaning up resources");
    fclose(outfile);
    free(upper_HALO);
    free(lower_HALO);
    free_grid(output_grid);
    free_grid(local_grid);

    LOG("Exiting simulate");
}



/*
 * AVX2: compute mask for 3 <= neighbor_count <= 5.
 * Returns 0x01 per byte if cell should be alive, otherwise 0x00.
 *
 * Must not be called on a boundary element.
 * Arrays must be length ≥ 34 and passed as arr + 1.
 */
__m256i determine_state256(const char* lower_arr, const char* middle_arr, const char* upper_arr)
{
    LOG("Entering determine_state256");

    __m256i sum = _mm256_setzero_si256();

    LOG("Load vertical neighbors");
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)upper_arr));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)lower_arr));

    LOG("Load horizontal neighbors");
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(middle_arr - 1)));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(middle_arr + 1)));

    LOG("Load diagonal neighbors");
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(upper_arr - 1)));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(upper_arr + 1)));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(lower_arr - 1)));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(lower_arr + 1)));

    LOG("Applying rule");

    const __m256i two = _mm256_set1_epi8(2);
    const __m256i six = _mm256_set1_epi8(6);
    const __m256i one = _mm256_set1_epi8(1);

    __m256i ge3 = _mm256_cmpgt_epi8(sum, two);  // sum >= 3
    __m256i le5 = _mm256_cmpgt_epi8(six, sum);  // sum <= 5

    LOG("Exiting determine_state256");
    return _mm256_and_si256(_mm256_and_si256(ge3, le5), one);
}

/*
 * SSE2: compute mask for 3 <= neighbor_count <= 5.
 * Returns 0x01 per byte if cell should be alive, otherwise 0x00.
 *
 * Must not be called on a boundary element.
 * Arrays must be length >= 18 and passed as arr + 1.
 */
__m128i determine_state128(const char* lower_arr, const char* middle_arr, const char* upper_arr)
{
    LOG("Entering determine_state128");

    __m128i sum = _mm_setzero_si128();

    LOG("Load vertical neighbors");
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)upper_arr));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)lower_arr));

    LOG("Load horizontal neighbors");
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(middle_arr - 1)));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(middle_arr + 1)));

    LOG("Load diagonal neighbors");
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(upper_arr - 1)));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(upper_arr + 1)));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(lower_arr - 1)));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(lower_arr + 1)));

    LOG("Applying rule");

    const __m128i two = _mm_set1_epi8(2);
    const __m128i six = _mm_set1_epi8(6);
    const __m128i one = _mm_set1_epi8(1);

    __m128i ge3 = _mm_cmpgt_epi8(sum, two);  // sum >= 3
    __m128i le5 = _mm_cmpgt_epi8(six, sum);  // sum <= 5

    LOG("Exiting determine_state128");
    return _mm_and_si128(_mm_and_si128(ge3, le5), one);
}

/* Scalar version: compute next state for a single cell.
 *
 * Assumptions:
 * - middle_arr points to the current cell
 * - arr[-1] and arr[+1] must be valid
 * - upper and lower rows must exist
 */
char determine_state1(
    const char* lower_arr,
    const char* middle_arr,
    const char* upper_arr)
{
    char neighbors =
        upper_arr[-1] + upper_arr[0] + upper_arr[1] +
        middle_arr[-1]              + middle_arr[1] +
        lower_arr[-1] + lower_arr[0] + lower_arr[1];

    return (neighbors >= 3 && neighbors <= 5) ? (char)0x01 : (char)0x00;
}

/* 
 * Safe scalar version: compute the next state for a single cell
 * on a toroidal grid (wraps around left/right edges).
 *
 * grid: pointer to the grid structure
 * row, col: coordinates of the cell
 */
char determine_state1_manual_rows(char* upper_row, char* middle_row, char* lower_row, int ncols, int col)
{
    int left  = (col == 0) ? ncols - 1 : col - 1;
    int right = (col == ncols - 1) ? 0 : col + 1;

    char neighbors =
        upper_row[left]   + upper_row[col]   + upper_row[right] +
        middle_row[left]                  + middle_row[right] +
        lower_row[left]   + lower_row[col]   + lower_row[right];

    return (neighbors >= 3 && neighbors <= 5) ? (char)0x01 : (char)0x00;
}

void calculate_row(
    char* upper_row,
    char* middle_row,
    char* lower_row,
    char* output_row,
    int ncols,
    int blocks_256,
    int blocks_128,
    int blocks_1
) {
    LOG("Processing manual row");

    output_row[0] = determine_state1_manual_rows(
        upper_row,
        middle_row,
        lower_row,
        ncols,
        0
    );

    output_row[ncols - 1] = determine_state1_manual_rows(
        upper_row,
        middle_row,
        lower_row,
        ncols,
        ncols-1
    );

    upper_row  += 1;
    middle_row += 1;
    lower_row  += 1;
    output_row += 1;


    LOG("Processing 256-bit blocks");
    for(int i = 0; i < blocks_256; i++) {
        _mm256_storeu_si256(
            (__m256i*)output_row, 
            determine_state256(
                lower_row,
                middle_row,
                upper_row
            )
        );
        upper_row += 32;
        middle_row += 32;
        lower_row += 32;
        output_row += 32;
    }

    LOG("Processing 128-bit blocks");
    for(int i = 0; i < blocks_128; i++) {
        _mm_storeu_si128(
            (__m128i*)output_row, 
            determine_state128(
                lower_row,
                middle_row,
                upper_row
            )
        );
        upper_row += 16;
        middle_row += 16;
        lower_row += 16;
        output_row += 16;
    }

    LOG("Processing remaining scalar cells");
    for(int i = 0; i < blocks_1; i++) {
        *output_row = determine_state1(
            lower_row,
            middle_row,
            upper_row
        );
        upper_row += 1;
        middle_row += 1;
        lower_row += 1;
        output_row += 1;
    }
}