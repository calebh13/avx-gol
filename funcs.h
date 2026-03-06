#ifndef FUNCS
#define FUNCS

#include <mpi.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>

void GenerateInitialGoL(int n, int rows, char** local_grid);
void free_grid(int n, int rows, int** local_grid);

#endif