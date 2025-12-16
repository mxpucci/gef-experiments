#!/bin/bash

# Check if input directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_with_bin_files>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="benchmark_results"

# Locate the benchmark executable
if [ -f "./build/LosslessBenchmarkFull" ]; then
    BENCH_EXE="./build/LosslessBenchmarkFull"
    echo "Using: LosslessBenchmarkFull (with Squash support)"
elif [ -f "./build/LosslessBenchmark" ]; then
    BENCH_EXE="./build/LosslessBenchmark"
    echo "Using: LosslessBenchmark (Minimal)"
else
    echo "Error: Could not find 'LosslessBenchmark' or 'LosslessBenchmarkFull' in ./build/"
    echo "Please build the project first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Loop through all .bin files in the directory
for FILE in "$INPUT_DIR"/*.bin; do
    # Check if file exists (in case glob fails)
    if [ ! -e "$FILE" ]; then
        echo "No .bin files found in $INPUT_DIR"
        exit 0
    fi

    BASENAME=$(basename "$FILE")
    FILENAME="${BASENAME%.*}"
    OUTPUT_FILE="$OUTPUT_DIR/${FILENAME}.txt"

    echo "Running benchmark for: $BASENAME"
    echo "Output saving to: $OUTPUT_FILE"

    # Run the benchmark
    # -o specifies the output file
    # We capture stdout/stderr to terminal for progress monitoring, 
    # but the tool itself writes the CSV results to the -o file.
    "$BENCH_EXE" -o "$OUTPUT_FILE" "$FILE"
done

echo "All benchmarks completed. Results stored in '$OUTPUT_DIR/'"

