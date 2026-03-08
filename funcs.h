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

__m256i determine_state256(
    const char* lower_arr,
    const char* middle_arr,
    const char* upper_arr);

#endif