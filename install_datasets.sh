#!/bin/bash

# Pull the datasets using git lfs
# Note: Ensure the files are tracked in their new location if they were moved.
echo "Pulling datasets from Git LFS..."
git lfs pull --include="datasets/datasets_archive.*"

# Check if the pull was successful or if files exist
if [ ! -f "datasets/datasets_archive.zip" ]; then
    echo "Error: datasets/datasets_archive.zip not found. LFS pull might have failed or files are not in the expected location."
    exit 1
fi

echo "Extracting datasets..."

# Try using 7z first as it handles split archives reliably
if command -v 7z &> /dev/null; then
    # -y assumes yes on all queries
    7z x datasets/datasets_archive.zip -y
elif command -v unzip &> /dev/null; then
    unzip -o datasets/datasets_archive.zip
else
    echo "Error: Neither 7z nor unzip found. Please install one of them."
    exit 1
fi

echo "Datasets installed successfully in datasets/"
