#!/bin/bash
set -uo pipefail
export LC_ALL=C LANG=C
run_dir="$HOME/gef-benchmark-runs/20261104-buff-leco-native"
base_dir="$HOME/gef-benchmark-runs/20261104-all-strict-input"
extra_root=/dev/shm/testPF-gef-extra-20261104
export PATH="$HOME/.local/share/gef-toolchain/envs/build/bin:$extra_root/deps/bin:$PATH"
export LD_LIBRARY_PATH="$extra_root/deps/lib:$HOME/.local/share/gef-toolchain/envs/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo 'Waiting for full-suite completion; this rerun will supersede only BUFF and LeCo.'
while [ ! -f "$base_dir/archive.sha256" ]; do sleep 10; done
test "$(cat "$run_dir/build.exit")" = 0 || exit 1
/usr/local/bin/python3.7 "$run_dir/run.py" > "$run_dir/output.txt" 2>&1
runner_rc=$?; echo "$runner_rc" > "$run_dir/supervisor.exit"
/usr/local/bin/python3.7 "$run_dir/audit.py" > "$run_dir/audit-output.txt" 2>&1
echo "$?" > "$run_dir/audit.exit"
tar --exclude=20261104-buff-leco-native/LosslessBenchmark --exclude=20261104-buff-leco-native/leco-native-test -czf "$HOME/gef-benchmark-runs/20261104-buff-leco-native.tar.gz" -C "$HOME/gef-benchmark-runs" 20261104-buff-leco-native
sha256sum "$HOME/gef-benchmark-runs/20261104-buff-leco-native.tar.gz" > "$run_dir/archive.sha256"
exit "$runner_rc"
