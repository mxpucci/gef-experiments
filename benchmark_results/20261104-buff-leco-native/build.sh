#!/bin/bash
set -euo pipefail
export LC_ALL=C LANG=C
run_dir="$HOME/gef-benchmark-runs/20261104-buff-leco-native"
extra_root=/dev/shm/testPF-gef-extra-20261104
export PATH="$HOME/.local/share/gef-toolchain/envs/build/bin:$extra_root/deps/bin:$PATH"
export LD_LIBRARY_PATH="$extra_root/deps/lib:$HOME/.local/share/gef-toolchain/envs/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$HOME/gef-experiments"
cmake --build build --target LosslessBenchmarkFull -j4 > "$run_dir/build.txt" 2>&1
CARGO_TARGET_DIR="$PWD/build/buff-ffi-cargo-target" cargo test --release --manifest-path lib/buff-ffi/Cargo.toml > "$run_dir/buff-tests.txt" 2>&1
"$HOME/.local/share/gef-toolchain/envs/build/bin/x86_64-conda-linux-gnu-g++" -O3 -DNDEBUG -march=native -std=c++17 -Ibenchmark tests/leco_native.cpp build/leco_benchmark_clang.o -L"$extra_root/deps/lib" -lgmp -static-libgcc -static-libstdc++ -o "$run_dir/leco-native-test"
"$run_dir/leco-native-test" /dev/shm/testPF-gef-20261103/extracted/datasets/*.bin > "$run_dir/leco-tests.txt" 2>&1
cp build/LosslessBenchmarkFull "$run_dir/LosslessBenchmark"
sha256sum "$run_dir/LosslessBenchmark" > "$run_dir/binary.sha256"
git diff > "$run_dir/source.patch"
git rev-parse HEAD > "$run_dir/source-base-commit.txt"
echo 0 > "$run_dir/build.exit"
