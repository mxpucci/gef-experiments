#!/bin/bash
set -uo pipefail
export LC_ALL=C LANG=C
run_dir="$HOME/gef-benchmark-runs/20261104-remaining"
extra_root=/dev/shm/testPF-gef-extra-20261104
export PATH="$HOME/.local/share/gef-toolchain/envs/build/bin:$extra_root/deps/bin:$PATH"
export LD_LIBRARY_PATH="$extra_root/deps/lib:$HOME/.local/share/gef-toolchain/envs/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SQUASH_PLUGINS="$extra_root/squash-install/lib64/squash/0.7/plugins:$extra_root/squash-install/lib/squash/0.7/plugins"
test "$(cat "$run_dir/build.exit")" = 0 || exit 1
/usr/local/bin/python3.7 "$run_dir/run.py" > "$run_dir/output.txt" 2>&1
runner_rc=$?
echo "$runner_rc" > "$run_dir/supervisor.exit"
/usr/local/bin/python3.7 "$run_dir/audit.py" > "$run_dir/audit-output.txt" 2>&1
echo "$?" > "$run_dir/audit.exit"
tar --exclude=20261104-remaining/LosslessBenchmark -czf "$HOME/gef-benchmark-runs/20261104-remaining.tar.gz" -C "$HOME/gef-benchmark-runs" 20261104-remaining
sha256sum "$HOME/gef-benchmark-runs/20261104-remaining.tar.gz" > "$run_dir/archive.sha256"
exit "$runner_rc"
