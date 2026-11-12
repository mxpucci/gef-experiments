#!/bin/bash
set -euo pipefail
export LC_ALL=C
export PATH="$HOME/.local/share/gef-toolchain/envs/build/bin:$PATH"
RUN="$HOME/gef-benchmark-runs/20261103-float32-warnings"
exec > >(tee "$RUN/build.txt") 2>&1
trap 'rc=$?; echo "$rc" > "$RUN/build.exit"' EXIT
cd "$HOME/gef-experiments"
git rev-parse HEAD > "$RUN/source-base-commit.txt"
sha256sum "$RUN/source.patch" > "$RUN/source-patch.sha256"
cmake --build build -j4
ctest --test-dir build --output-on-failure
cp build/LosslessBenchmark "$RUN/LosslessBenchmark"
cp build/Testing/Temporary/LastTest.log "$RUN/build-tests.txt"
sha256sum "$RUN/LosslessBenchmark" > "$RUN/binary.sha256"
