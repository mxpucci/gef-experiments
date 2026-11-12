#!/bin/bash
set -euo pipefail
export LC_ALL=C LANG=C
export PATH="$HOME/.local/share/gef-toolchain/envs/build/bin:$PATH"
cd "$HOME/gef-experiments"
run_dir="$HOME/gef-benchmark-runs/20261103-native-access"
cmake --build build -j4 > "$run_dir/build.txt" 2>&1
ctest --test-dir build --output-on-failure > "$run_dir/tests.txt" 2>&1
echo 0 > "$run_dir/build.exit"
/usr/local/bin/python3.7 "$run_dir/run.py" > "$run_dir/output.txt" 2>&1
