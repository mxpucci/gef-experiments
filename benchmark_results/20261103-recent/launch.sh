#!/bin/bash
set -euo pipefail
export LC_ALL=C
RUN="$HOME/gef-benchmark-runs/20261103-recent"
cd "$RUN"
while [ ! -f preparation.exit ]; do
  if ! tmux has-session -t gef-prepare 2>/dev/null; then
    echo 'Preparation session disappeared without an exit record.' > output.txt
    exit 2
  fi
  sleep 5
done
if [ "$(cat preparation.exit)" != 0 ]; then
  echo 'Preparation failed; see preparation.txt' > output.txt
  exit 2
fi
/usr/local/bin/python3.7 -u run.py 2>&1 | tee output.txt
