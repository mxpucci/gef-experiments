#!/bin/bash

# Configuration
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
PLOT_SCRIPT="scripts/pareto_plots.py"
# Default input/output if not provided as arguments
DEFAULT_CSV="benchmark_results/results.csv"
DEFAULT_OUT_DIR="plots"

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
CSV_FILE="${1:-$DEFAULT_CSV}"
OUT_DIR="${2:-$DEFAULT_OUT_DIR}"

if [ ! -f "$CSV_FILE" ]; then
    # Try to find any CSV in benchmark_results if default doesn't exist
    CSV_FILE=$(find benchmark_results -name "*.csv" | head -n 1)
    if [ -z "$CSV_FILE" ]; then
        echo "Error: No CSV file found to process."
        echo "Usage: ./generate_plots.sh [csv_file] [output_dir]"
        exit 1
    fi
    echo "Default CSV not found, using found CSV: $CSV_FILE"
fi

# Run the plotting script
echo "Generating plots from $CSV_FILE to $OUT_DIR..."
python3 "$PLOT_SCRIPT" "$CSV_FILE" "$OUT_DIR"

echo "Done."










