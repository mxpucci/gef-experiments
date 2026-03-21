#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="${REPO_ROOT}/datasets/inverted-index"
OUTPUT_DIR="${REPO_ROOT}/datasets/inverted-index-unpacked"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    echo "Run: git submodule update --init datasets/inverted-index"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

for zipfile in "$DATASET_DIR"/*.zip; do
    [ -f "$zipfile" ] || continue
    basename=$(basename "$zipfile" .zip)
    dest="$OUTPUT_DIR/$basename"

    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "Skipping $basename (already unpacked)"
        continue
    fi

    echo "Unpacking $basename..."
    mkdir -p "$dest"
    unzip -q -o "$zipfile" -d "$dest"
done

echo ""
echo "All datasets unpacked to: $OUTPUT_DIR"
echo ""
for dir in "$OUTPUT_DIR"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    count=$(find "$dir" -name "*.txt" | wc -l | tr -d ' ')
    echo "  $name: $count posting lists"
done
