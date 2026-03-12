#include "gol.h"
#include <mpi.h>

// If true, p0 will read each process's output file in order
// Should only be set for debugging.
#define PRINT_FULL_MATRIX 0

double comm_time = 0;

struct Grid {
    char** grid;
    size_t n;
    size_t rows;
};

// Allocates a grid of bytes to the specified size
Grid* init_grid(int n, int rows) {
    Grid* grid = malloc(sizeof(Grid));
    if (!grid) return NULL;

    grid->grid = malloc(rows * sizeof(char*));
    if (!grid->grid) return NULL;

    for (int i = 0; i < rows; i++) {
        grid->grid[i] = malloc(n * sizeof(char));
        if (!grid->grid[i]) return NULL;
    }

    grid->n = n;
    grid->rows = rows;
    return grid;
}

// Prints the input grid to the specified file. Uses stdout if no output file specified
void print_grid(const Grid* g, FILE* out) {
    if (!out) out = stdout;  // default to stdout if NULL
    for (size_t r = 0; r < g->rows; r++) {
        for (size_t c = 0; c < g->n; c++) {
            fprintf(out, "%d ", g->grid[r][c]);
        }

        fprintf(out, "\n");
    }

    fprintf(out, "\n");
    fflush(out);
}

// Reads every process's output file, and prints it to stdout.
void DisplayGoL(int n, int generation) {
    char filename[64];
    size_t linesize = (n+1) * 2 * sizeof(char);
    char* line = malloc(linesize);

    for(int r = 0; r < p; r++) {
        sprintf(filename, OUTFILE_FORMAT, r);
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
}

// Creates random grid seeds and sends them to each proc
void scatter_seeds() {
    unsigned int seed;
    double t0;

    if (rank == 0) {
        unsigned int *seeds = malloc(p * sizeof(unsigned int));
        srand(time(NULL));
        for (int i = 0; i < p; i++) {
            seeds[i] = rand() % BIGPRIME + 1;
        }

        t0 = MPI_Wtime();
        MPI_Scatter(seeds, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - t0;

        free(seeds);
    } else {

        t0 = MPI_Wtime();
        MPI_Scatter(NULL, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - t0;

    }

    srand(seed);
}

// Fills a grid with random 0's or 1's
void GenerateInitialGoL(Grid* local_grid) {
    scatter_seeds();
    for(int i = 0; i < local_grid->rows; i++) {
        for(int j = 0; j < local_grid->n; j++){
            local_grid->grid[i][j] = rand() % 2 == 0;
        }
    }
}

// Free's the input exam
void free_grid(Grid* local_grid) {
    for(int i = 0; i < local_grid->rows; i++){
        free(local_grid->grid[i]);
    }

    free(local_grid->grid);
    free(local_grid);
}


// User must call MPI_Wait on send_req and recv_req before attempting to use the relevant buffers
// This is not critical for the send buffer (so long as it isn't freed), but is extremely important for the recv buffer
static void sendLowerRecvUpper(Grid* local_grid, char* upper_row, MPI_Request* send_req, MPI_Request* recv_req) {
    int send_target = (rank + 1) % p;
    int recv_source = (rank + p - 1) % p;

    double t0 = MPI_Wtime();

    MPI_Isend(
        local_grid->grid[local_grid->rows - 1],
        local_grid->n,
        MPI_CHAR,
        send_target,
        0,
        MPI_COMM_WORLD,
        send_req
    );

    MPI_Irecv(
        upper_row,
        local_grid->n,
        MPI_CHAR,
        recv_source,
        MPI_ANY_TAG,
        MPI_COMM_WORLD,
        recv_req
    );

    comm_time += MPI_Wtime() - t0;
}

// User must call MPI_Wait on send_req and recv_req before attempting to use the relevant buffers
// This is not critical for the send buffer (so long as it isn't freed), but is extremely important for the recv buffer
static void sendUpperRecvLower(Grid* local_grid, char* lower_row, MPI_Request* send_req, MPI_Request* recv_req) {
    int send_target = (rank + p - 1) % p;
    int recv_source = (rank + 1) % p;

    double t0 = MPI_Wtime();

    MPI_Isend(
        local_grid->grid[0],
        local_grid->n,
        MPI_CHAR,
        send_target,
        0,
        MPI_COMM_WORLD,
        send_req
    );

    MPI_Irecv(
        lower_row,
        local_grid->n,
        MPI_CHAR,
        recv_source,
        MPI_ANY_TAG,
        MPI_COMM_WORLD,
        recv_req
    );

    comm_time += MPI_Wtime() - t0;
}

// Main logic loop. Simulates g generations using local_grid as an input starting state //
void simulate(Grid* local_grid, int g, int x) {
    assert(g >= 0);

    int blocks_512 = (local_grid->n - 2) / 64;
    int remainder = (local_grid->n - 2) % 64;
    int blocks_256 = remainder / 32;
    remainder %= 32;
    int blocks_128 = remainder / 16;
    int blocks_1 = remainder % 16;

    char* upper_HALO = malloc(sizeof(char) * local_grid->n);
    char* lower_HALO = malloc(sizeof(char) * local_grid->n);
    MPI_Request upper_send, upper_recv, lower_send, lower_recv;

    Grid* output_grid = init_grid(local_grid->n, local_grid->rows);

    char outfile_name[64];
    sprintf(outfile_name, OUTFILE_FORMAT, rank);
    FILE* outfile = fopen(outfile_name, "w");

    if (x > 0) {
        print_grid(local_grid, outfile);
        if (PRINT_FULL_MATRIX) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                DisplayGoL(local_grid->n, 0);
            }
        }
    }

    for(int current_gen = 1; current_gen <= g; current_gen++) {
        if(rank == 0){
            printf("Gen: %d\n", current_gen);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        sendLowerRecvUpper(local_grid, upper_HALO, &lower_send, &upper_recv);
        sendUpperRecvLower(local_grid, lower_HALO, &upper_send, &lower_recv);

        for(int row = 1; row < local_grid->rows - 1; row++) {
            calculate_row(
                local_grid->grid[row - 1],
                local_grid->grid[row],
                local_grid->grid[row + 1],
                output_grid->grid[row],
                local_grid->n,
                blocks_512,
                blocks_256,
                blocks_128,
                blocks_1
            );
        }

        double t0 = MPI_Wtime();

        MPI_Wait(&upper_send, MPI_STATUS_IGNORE);
        MPI_Wait(&lower_send, MPI_STATUS_IGNORE);
        MPI_Wait(&upper_recv, MPI_STATUS_IGNORE);
        MPI_Wait(&lower_recv, MPI_STATUS_IGNORE);

        comm_time += MPI_Wtime() - t0;

        calculate_row(
            upper_HALO,
            local_grid->grid[0],
            local_grid->grid[1],
            output_grid->grid[0],
            local_grid->n,
            blocks_512,
            blocks_256,
            blocks_128,
            blocks_1
        );

        calculate_row(
            local_grid->grid[local_grid->rows-2],
            local_grid->grid[local_grid->rows-1],
            lower_HALO,
            output_grid->grid[local_grid->rows-1],
            local_grid->n,
            blocks_512,
            blocks_256,
            blocks_128,
            blocks_1
        );

        Grid* temp = local_grid;
        local_grid = output_grid;
        output_grid = temp;

        if (x > 0 && current_gen % x == 0) {
            print_grid(local_grid, outfile);
            if (PRINT_FULL_MATRIX) {
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) {
                    DisplayGoL(local_grid->n, current_gen);
                }
            }
        }
    }

    fclose(outfile);
    free(upper_HALO);
    free(lower_HALO);
    free_grid(output_grid);
    free_grid(local_grid);
}

/* Returns 0x01 per byte if cell should be alive, otherwise 0x00.
 *
 * Must not be called on a boundary element.
 * Arrays must be length ≥ 66 and passed as arr + 1.
 */
__m512i determine_state512(const char* lower_arr, const char* middle_arr, const char* upper_arr) {
    __m512i sum = _mm512_setzero_si512();

    sum = _mm512_add_epi8(sum, _mm512_loadu_si512((const void*)upper_arr));
    sum = _mm512_add_epi8(sum, _mm512_loadu_si512((const void*)lower_arr));

    sum = _mm512_add_epi8(sum, _mm512_loadu_si512((const void*)(middle_arr - 1)));
    sum = _mm512_add_epi8(sum, _mm512_loadu_si512((const void*)(middle_arr + 1)));

    sum = _mm512_add_epi8(sum, _mm512_loadu_si512((const void*)(upper_arr - 1)));
    sum = _mm512_add_epi8(sum, _mm512_loadu_si512((const void*)(upper_arr + 1)));
    sum = _mm512_add_epi8(sum, _mm512_loadu_si512((const void*)(lower_arr - 1)));
    sum = _mm512_add_epi8(sum, _mm512_loadu_si512((const void*)(lower_arr + 1)));

    const __m512i two = _mm512_set1_epi8(2);
    const __m512i six = _mm512_set1_epi8(6);
    const __m512i one = _mm512_set1_epi8(1);

    __mmask64 ge3 = _mm512_cmpgt_epi8_mask(sum, two); // sum >= 3
    __mmask64 le5 = _mm512_cmpgt_epi8_mask(six, sum); // sum <= 5
    __mmask64 alive_mask = ge3 & le5;

    return _mm512_maskz_mov_epi8(alive_mask, one);
}

/* Returns 0x01 per byte if cell should be alive, otherwise 0x00.
 *
 * Must not be called on a boundary element.
 * Arrays must be length ≥ 34 and passed as arr + 1. */
__m256i determine_state256(const char* lower_arr, const char* middle_arr, const char* upper_arr) {
    __m256i sum = _mm256_setzero_si256();

    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)upper_arr));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)lower_arr));

    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(middle_arr - 1)));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(middle_arr + 1)));

    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(upper_arr - 1)));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(upper_arr + 1)));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(lower_arr - 1)));
    sum = _mm256_add_epi8(sum, _mm256_loadu_si256((const __m256i*)(lower_arr + 1)));

    const __m256i two = _mm256_set1_epi8(2);
    const __m256i six = _mm256_set1_epi8(6);
    const __m256i one = _mm256_set1_epi8(1);

    __m256i ge3 = _mm256_cmpgt_epi8(sum, two);  // sum >= 3
    __m256i le5 = _mm256_cmpgt_epi8(six, sum);  // sum <= 5

    return _mm256_and_si256(_mm256_and_si256(ge3, le5), one);
}

/* Returns 0x01 per byte if cell should be alive, otherwise 0x00.
 *
 * Must not be called on a boundary element.
 * Arrays must be length >= 18 and passed as arr + 1. */
__m128i determine_state128(const char* lower_arr, const char* middle_arr, const char* upper_arr) {
    __m128i sum = _mm_setzero_si128();

    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)upper_arr));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)lower_arr));

    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(middle_arr - 1)));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(middle_arr + 1)));

    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(upper_arr - 1)));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(upper_arr + 1)));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(lower_arr - 1)));
    sum = _mm_add_epi8(sum, _mm_loadu_si128((const __m128i*)(lower_arr + 1)));

    const __m128i two = _mm_set1_epi8(2);
    const __m128i six = _mm_set1_epi8(6);
    const __m128i one = _mm_set1_epi8(1);

    __m128i ge3 = _mm_cmpgt_epi8(sum, two);  // sum >= 3
    __m128i le5 = _mm_cmpgt_epi8(six, sum);  // sum <= 5

    return _mm_and_si128(_mm_and_si128(ge3, le5), one);
}

/* Returns 0x01 per byte if cell should be alive, otherwise 0x00.
 *
 * Must not be called on a boundary element.
 * Arrays must be length >= 1 and passed as arr + 1. */
char determine_state1(
    const char* lower_arr,
    const char* middle_arr,
    const char* upper_arr) {
    char neighbors =
        upper_arr[-1] + upper_arr[0] + upper_arr[1] +
        middle_arr[-1]              + middle_arr[1] +
        lower_arr[-1] + lower_arr[0] + lower_arr[1];

    return (neighbors >= 3 && neighbors <= 5) ? (char)0x01 : (char)0x00;
}

/* Returns 0x01 per byte if cell should be alive, otherwise 0x00.
 *
 * Can be called on a boundary element.
 * Arrays must be length >= 1 and passed as arr + 1. */
char determine_state1_manual_rows(char* upper_row, char* middle_row, char* lower_row, int ncols, int col) {
    int left  = (col == 0) ? ncols - 1 : col - 1;
    int right = (col == ncols - 1) ? 0 : col + 1;

    char neighbors =
        upper_row[left]   + upper_row[col]   + upper_row[right] +
        middle_row[left]                  + middle_row[right] +
        lower_row[left]   + lower_row[col]   + lower_row[right];

    return (neighbors >= 3 && neighbors <= 5) ? (char)0x01 : (char)0x00;
}

/* Manually calculates a full row and puts the resulting cell updates into output_row */
#include <immintrin.h>

void calculate_row(
    char* upper_row,
    char* middle_row,
    char* lower_row,
    char* output_row,
    int ncols,
    int blocks_512,
    int blocks_256,
    int blocks_128,
    int blocks_1) {

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
        ncols - 1
    );

    upper_row  += 1;
    middle_row += 1;
    lower_row  += 1;
    output_row += 1;

    for (int i = 0; i < blocks_512; i++) {
        _mm512_storeu_si512(
            (void*)output_row,
            determine_state512(
                lower_row,
                middle_row,
                upper_row
            )
        );

        upper_row  += 64;
        middle_row += 64;
        lower_row  += 64;
        output_row += 64;
    }

    for (int i = 0; i < blocks_256; i++) {
        _mm256_storeu_si256(
            (__m256i*)output_row,
            determine_state256(
                lower_row,
                middle_row,
                upper_row
            )
        );

        upper_row  += 32;
        middle_row += 32;
        lower_row  += 32;
        output_row += 32;
    }

    for (int i = 0; i < blocks_128; i++) {
        _mm_storeu_si128(
            (__m128i*)output_row,
            determine_state128(
                lower_row,
                middle_row,
                upper_row
            )
        );

        upper_row  += 16;
        middle_row += 16;
        lower_row  += 16;
        output_row += 16;
    }

    for (int i = 0; i < blocks_1; i++) {
        *output_row = determine_state1(
            lower_row,
            middle_row,
            upper_row
        );

        upper_row  += 1;
        middle_row += 1;
        lower_row  += 1;
        output_row += 1;
    }
}