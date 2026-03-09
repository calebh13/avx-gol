# AVX GoL
C implementation of John Conway's Game of Life using MPI for parallelization and AVX2 intrinsics to vectorize certain operations.

Currently, the game grid is stored as a byte-array, so AVX256 operates on 32 cells at a time. This may be converted to a bit-packed representation in the future depending on profiling results.
