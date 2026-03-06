#include "funcs.h"

#define BIGPRIME 2147483647UL
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

void GenerateInitialGoL(int n, int rows, int** local_grid) {
    
}
