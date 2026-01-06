#!/bin/bash

# Configuration
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
TABLE_SCRIPT="scripts/tables.py"
# Default input/output if not provided as arguments
DEFAULT_CSV="benchmark_results/dataset_normalized_optimized.csv"
DEFAULT_OUT_DIR="tables"
DEFAULT_THRESHOLD="0.35"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing/Updating requirements..."
    pip install -q -r "$REQUIREMENTS_FILE"
else
    echo "Warning: $REQUIREMENTS_FILE not found."
fi

# Determine arguments
# Usage: ./generate_tables.sh [csv_file] [output_dir] [threshold] [--compressors "list"] [--include-approx]

POSITIONAL_ARGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --compressors)
            EXTRA_ARGS+=("--compressors" "$2")
            shift 2
            ;;
        --include-approx)
            EXTRA_ARGS+=("--include-approx")
            shift
            ;;
        --datasets)
            EXTRA_ARGS+=("--datasets" "$2")
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

CSV_FILE="${POSITIONAL_ARGS[0]:-$DEFAULT_CSV}"
OUT_DIR="${POSITIONAL_ARGS[1]:-$DEFAULT_OUT_DIR}"
THRESHOLD="${POSITIONAL_ARGS[2]:-$DEFAULT_THRESHOLD}"

if [ ! -f "$CSV_FILE" ]; then
    # Try to find any CSV in benchmark_results if default doesn't exist
    CSV_FILE=$(find benchmark_results -name "*.csv" | head -n 1)
    if [ -z "$CSV_FILE" ]; then
        echo "Error: No CSV file found to process."
        echo "Usage: ./generate_tables.sh [csv_file] [output_dir] [threshold] [--compressors list] [--include-approx]"
        exit 1
    fi
    echo "Default CSV not found, using found CSV: $CSV_FILE"
fi

# Run the table script
echo "Generating tables from $CSV_FILE with threshold $THRESHOLD to $OUT_DIR..."
python3 "$TABLE_SCRIPT" "$CSV_FILE" "$THRESHOLD" "$OUT_DIR" "${EXTRA_ARGS[@]}"

echo "Done."

