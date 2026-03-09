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

void GenerateInitialGoL(int n, int rows, char** local_grid);
void free_grid(int n, int rows, char** local_grid);

__m256i calculate_neighbors256(
    const char* lower_arr,
    const char* middle_arr,
    const char* upper_arr);

__m256i determine_state256(
    __m256i neighbor_counts
);

__m128i calculate_neighbors128(
    const char* lower_arr, 
    const char* middle_arr, 
    const char* upper_arr);
    
__m128i determine_state128(__m128i neighbor_counts);

#endif