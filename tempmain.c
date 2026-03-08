#include "funcs.h"

#define SIZE 34

void fill_random01(char *arr) {
    for (int i = 0; i < SIZE; i++) {
        arr[i] = rand() % 2;   // 0 or 1
    }
}


int main(int argc, char* argv[]) {
    printf("Hello World\n");
    
    MPI_Init(&argc, &argv);

    const int n = 16;
    const int G = 2;

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    assert(n > p && n % p == 0);
    char upper[SIZE];
    char middle[SIZE];
    char lower[SIZE];

    srand(12345);

    fill_random01(upper);
    fill_random01(middle);
    fill_random01(lower);

    for(int i = 0; i < SIZE; i++){
        printf("%c", upper[i] + 48);
    }

    printf("\n");
    for(int i = 0; i < SIZE; i++){
        printf("%c", middle[i] + 48);
    }

    printf("\n");
    for(int i = 0; i < SIZE; i++){
        printf("%c", lower[i] + 48);
    }

    printf("\n");

    __m256i res = calculate_neighbors256(lower + 1, middle + 1, upper + 1);

    char* sums = (char*) &res;

    printf("\n ");
    for(int i = 0; i < 32; i++){
        printf("%c", sums[i]  + 48);
    }
    printf("\n");

    // now we need to calculate the sun dog manually, I think?
    

    MPI_Finalize();
    return 0;
}