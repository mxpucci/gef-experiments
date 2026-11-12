#!/bin/bash
set -euo pipefail
export LC_ALL=C
RUN="$HOME/gef-benchmark-runs/20261103-recent"
STAGE=/dev/shm/testPF-gef-20261103
mkdir -p "$RUN" "$STAGE"
chmod 700 "$STAGE"
exec >> "$RUN/preparation.txt" 2>&1
trap 'rc=$?; echo "Preparation exit=$rc $(date -Is)"; echo "$rc" > "$RUN/preparation.exit"' EXIT
echo "Preparing datasets $(date -Is)"
cd "$STAGE"
BASE=https://media.githubusercontent.com/media/mxpucci/gef-experiments/0888e798ff2a691d1281c2d4b591a216c2a78eac/datasets
for part in datasets_archive.z01 datasets_archive.zip; do
  curl --fail --location --retry 5 --connect-timeout 30 --speed-limit 1024 --speed-time 120 --continue-at - --output "$part" "$BASE/$part"
done
sha256sum -c <<'CHECKSUMS'
ab7fec75f79863ed9f6e6e20362425490a1f9b6cfd0e50841ee4fbed7b9ec867  datasets_archive.z01
43dd1784c4f9bf6f1e21bdb0bb5761d62a72b17ab243a4eb116d6efb261df7ac  datasets_archive.zip
CHECKSUMS
zip -s 0 datasets_archive.zip --out datasets_complete.zip
unzip -l datasets_complete.zip > "$RUN/archive-inventory.txt"
mkdir -p extracted
unzip -o datasets_complete.zip -d extracted
echo "Dataset extraction complete $(date -Is)"
