#ifndef FUNCS
#define FUNCS

#include <mpi.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>
#include <assert.h>

extern int rank, p;
extern FILE* logfile;

#define LOG(...) \
do { \
    if(logfile){ \
        fprintf(logfile, "p%d: ", rank); \
        fprintf(logfile, __VA_ARGS__); \
        fprintf(logfile, "\n"); \
        fflush(logfile); \
    } \
} while(0)

#define BIGPRIME 85935431U
#define OUTFILE_FORMAT "p%dout"

// Opaque struct so caller cannot see internals
typedef struct Grid Grid;

Grid* init_grid(int n, int rows);
static void scatter_seeds();
void GenerateInitialGoL(Grid* local_grid);
static void sendLowerRecvUpper(Grid* local_grid, char* upper_row, MPI_Request* send_req, MPI_Request* recv_req);
static void sendUpperRecvLower(Grid* local_grid, int* lower_row, MPI_Request* send_req, MPI_Request* recv_req);
void simulate(Grid* local_grid, int g);
void free_grid(Grid* local_grid);

static __m256i calculate_neighbors256(
    const char* lower_arr,
    const char* middle_arr,
    const char* upper_arr);

static __m256i determine_state256(
    __m256i neighbor_counts
);

static __m128i calculate_neighbors128(
    const char* lower_arr, 
    const char* middle_arr, 
    const char* upper_arr);
    
static __m128i determine_state128(__m128i neighbor_counts);

#endif