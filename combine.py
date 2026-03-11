import sys

def combine_files(num_files, output_file="out.txt"):
    files = [open(f"p{i}out.txt", "r") for i in range(num_files)]

    try:
        with open(output_file, "w") as out:
            while True:
                finished = False

                for f in files:
                    while True:
                        line = f.readline()

                        if line == "":
                            finished = True
                            break

                        if line.strip() == "":
                            break  # end of this processor's chunk for this generation

                        out.write(line)

                    if finished:
                        break

                if finished:
                    break

                # separate generations in output
                out.write("\n")

    finally:
        for f in files:
            f.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python combine.py <num_files> {output_filename}")
        sys.exit(1)

    n = int(sys.argv[1])
    outfile = "out"
    if len(sys.argv) == 3:
        outfile = sys.argv[2]
    combine_files(n, outfile)