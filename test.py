import sys

def read_generations(filename):
    generations = []
    current = []

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if line == "":
                if current:
                    generations.append(current)
                    current = []
                continue

            row = list(map(int, line.split()))
            current.append(row)

    if current:
        generations.append(current)

    return generations


def neighbors(grid):
    n = len(grid)
    m = len(grid[0])
    counts = [[0]*m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            s = 0
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % n
                    nj = (j + dj) % m
                    s += grid[ni][nj]
            counts[i][j] = s

    return counts


def step(grid):
    n = len(grid)
    m = len(grid[0])
    counts = neighbors(grid)

    next_grid = [[0]*m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if 3 <= counts[i][j] <= 5:
                next_grid[i][j] = 1

    return next_grid


def print_grid(grid):
    for row in grid:
        print(" ".join(map(str,row)))


def check_generations(gens):
    for g in range(len(gens)-1):
        expected = step(gens[g])
        actual = gens[g+1]

        print(f"Checking generation {g} -> {g+1}")

        if expected == actual:
            print("OK\n")
        else:
            print("Mismatch!")
            print("\nExpected:")
            print_grid(expected)

            print("\nActual:")
            print_grid(actual)
            print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python check_life.py <file>")
        sys.exit(1)

    gens = read_generations(sys.argv[1])
    check_generations(gens)