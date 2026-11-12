#!/bin/bash
set -uo pipefail
export LC_ALL=C LANG=C
export PATH="$HOME/.local/share/gef-toolchain/envs/build/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/share/gef-toolchain/envs/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_dir="$HOME/gef-benchmark-runs/20261104-elf-strict-input"
cd "$HOME/gef-experiments"
cmake --build build --target LosslessBenchmark -j4 > "$run_dir/build.txt" 2>&1
rc=$?
echo "$rc" > "$run_dir/build.exit"
if [ "$rc" != 0 ]; then exit "$rc"; fi
cmake -S tests/recent -B build-elf-strict-tests -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$HOME/.local/share/gef-toolchain/envs/build/bin/x86_64-conda-linux-gnu-g++" -DCMAKE_C_COMPILER="$HOME/.local/share/gef-toolchain/envs/build/bin/x86_64-conda-linux-gnu-gcc" -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_CXX_FLAGS="-march=native -mfpmath=sse" -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" -DGEF_WITH_PCODEC=OFF -DGEF_WITH_FAUNA=OFF -DGEF_WITH_FLEA=OFF > "$run_dir/test-build.txt" 2>&1
if [ "$?" != 0 ]; then exit 2; fi
cmake --build build-elf-strict-tests --target benchmark_input_test elfstar_roundtrip selfstar_roundtrip recent_harness_test -j4 >> "$run_dir/test-build.txt" 2>&1
if [ "$?" != 0 ]; then exit 2; fi
ctest --test-dir build-elf-strict-tests -R '^(benchmark_input|elfstar_roundtrip|selfstar_roundtrip|recent_harness)$' --output-on-failure > "$run_dir/tests.txt" 2>&1
rc=$?
echo "$rc" > "$run_dir/tests.exit"
if [ "$rc" != 0 ]; then exit "$rc"; fi
/usr/local/bin/python3.7 "$run_dir/run.py" > "$run_dir/output.txt" 2>&1
rc=$?
echo "$rc" > "$run_dir/supervisor.exit"
exit "$rc"
