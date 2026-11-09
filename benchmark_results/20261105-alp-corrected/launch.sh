#!/bin/bash
set -euo pipefail
export LC_ALL=C LANG=C
extra_root=/dev/shm/testPF-gef-extra-20261104
export PATH="$HOME/.local/share/gef-toolchain/envs/build/bin:$extra_root/deps/bin:$PATH"
export LD_LIBRARY_PATH="$extra_root/deps/lib:$HOME/.local/share/gef-toolchain/envs/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_dir="$HOME/gef-experiments/benchmark_results/20261105-alp-corrected"
cd "$HOME/gef-experiments"
cmake --build build --target LosslessBenchmarkFull -j4 > "$run_dir/build.log" 2>&1
cp build/LosslessBenchmarkFull "$run_dir/LosslessBenchmark"
sha256sum "$run_dir/LosslessBenchmark" > "$run_dir/binary.sha256"
python3 "$run_dir/run.py"
