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
    
    # Setup library path for bundled libraries
    # Assuming 'build/lib' contains the libraries relative to the script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    LIB_DIR="$SCRIPT_DIR/build/lib"
    
    if [ -d "$LIB_DIR" ]; then
        echo "Using bundled Squash libraries from: $LIB_DIR"
        export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
    else
        echo "WARNING: Bundled 'lib' directory not found at $LIB_DIR"
        echo "If Squash is not installed system-wide, the benchmark will fail."
        echo "Please ensure you copied the 'build/lib' directory along with the executable."
    fi
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

