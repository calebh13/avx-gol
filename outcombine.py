import glob

output_file = "results.csv"
files = sorted(glob.glob("*.out"))

with open(output_file, "w") as outfile:
    for fname in files:
        with open(fname, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            data = lines[0]

            outfile.write(data + "\n")

print(f"Combined {len(files)} files into {output_file}")
