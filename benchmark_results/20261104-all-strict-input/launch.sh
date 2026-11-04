#!/bin/bash
set -uo pipefail
export LC_ALL=C LANG=C
run_dir="$HOME/gef-benchmark-runs/20261104-all-strict-input"
elf_dir="$HOME/gef-benchmark-runs/20261104-elf-strict-input"
extra_root=/dev/shm/testPF-gef-extra-20261104
export PATH="$HOME/.local/share/gef-toolchain/envs/build/bin:$extra_root/deps/bin:$PATH"
export LD_LIBRARY_PATH="$extra_root/deps/lib:$HOME/.local/share/gef-toolchain/envs/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SQUASH_PLUGINS="$extra_root/squash-install/lib64/squash/0.7/plugins:$extra_root/squash-install/lib/squash/0.7/plugins"
echo 'Waiting for the current Elf*/SElf* cases to finish before building/running the other 30 codecs.'
while [ ! -f "$elf_dir/runner.exit" ]; do sleep 5; done
cd "$HOME/gef-experiments"
cmake --build build --target LosslessBenchmarkFull -j4 > "$run_dir/build.txt" 2>&1
rc=$?; echo "$rc" > "$run_dir/build.exit"
if [ "$rc" != 0 ]; then exit "$rc"; fi
cmake -S tests/recent -B build-recent -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_EXE_LINKER_FLAGS="-no-pie -static-libgcc -static-libstdc++" -DGEF_WITH_PCODEC=ON -DGEF_WITH_FAUNA=ON -DGEF_WITH_FLEA=ON > "$run_dir/test-build.txt" 2>&1
rc=$?; if [ "$rc" != 0 ]; then echo "$rc" > "$run_dir/tests.exit"; exit "$rc"; fi
cmake --build build-recent -j4 >> "$run_dir/test-build.txt" 2>&1
rc=$?; if [ "$rc" != 0 ]; then echo "$rc" > "$run_dir/tests.exit"; exit "$rc"; fi
ctest --test-dir build-recent --output-on-failure > "$run_dir/tests.txt" 2>&1
rc=$?; echo "$rc" > "$run_dir/tests.exit"
if [ "$rc" != 0 ]; then exit "$rc"; fi
git rev-parse HEAD > "$run_dir/source-base-commit.txt"
git diff > "$run_dir/source.patch"
sha256sum benchmark/benchmark_input.cpp benchmark/benchmark_common.hpp cmake/BenchmarkInput.cmake cmake/RecentCompressors.cmake > "$run_dir/input-source.sha256"
/usr/local/bin/python3.7 "$run_dir/run.py" > "$run_dir/output.txt" 2>&1
runner_rc=$?; echo "$runner_rc" > "$run_dir/supervisor.exit"
/usr/local/bin/python3.7 "$run_dir/audit.py" > "$run_dir/audit-output.txt" 2>&1
echo "$?" > "$run_dir/audit.exit"
tar --exclude=20261104-all-strict-input/LosslessBenchmark -czf "$HOME/gef-benchmark-runs/20261104-all-strict-input.tar.gz" -C "$HOME/gef-benchmark-runs" 20261104-all-strict-input
sha256sum "$HOME/gef-benchmark-runs/20261104-all-strict-input.tar.gz" > "$run_dir/archive.sha256"
exit "$runner_rc"
