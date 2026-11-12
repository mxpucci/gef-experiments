STATUS: STOPPED/CANCELLED at user request on 2026-11-04. No automatic restart.
See ../RUN_STATUS.txt; scheduling descriptions below are historical.

Full-suite rerun authorized 2026-11-04 after discovering input-conversion error.
32 codecs x 16 full datasets = 512 measurements, split across two corrected runs:
- 20261104-elf-strict-input: Elf*/SElf*, 32 measurements, runs first.
- 20261104-all-strict-input: other 30 codecs, 480 measurements, queued afterward.
All prior measurements remain preserved. New results-all.txt contains only these
corrected runs; no old reciprocal-conversion results are used as replacements.

Scaled-integer to binary64 input conversion now uses strict division in a
non-LTO translation unit. Codec implementations and all existing settings remain
unchanged. Sequential isolated cases on CPU 2, full datasets, no case timeout.
Block size 1000 for existing external-block adapters; native configurations are
unchanged (Pcodec chunk/page 262144, Fauna batch 65536). Pcodec/Fauna access-method
and page-size concerns are not addressed by this rerun and remain qualified.
Float32 Fauna/Macaque conversion and legacy validation limitations remain visible.
Warnings retain metrics; individual hard failures do not stop the entire run.

Brotli/NeaTS are included again under the expanded user request. The previous
remaining-compressor run stays cancelled; its 366 completed cases are preserved.
The supervisor waits for current Elf*/SElf* benchmarks, builds the full target,
runs tests, executes all 480 additional cases, audits and combines both corrected
runs, and produces a logs archive (benchmark binaries remain on server).
Remote: ~/gef-benchmark-runs/20261104-all-strict-input
Tmux: gef-all-strict
Progress: output.txt, status.json; completion: runner.exit and archive.sha256.

STARTED: 2026-11-04T10:45:41.253188+00:00; full binary built and all 10 tests passed.
Local status/results files are snapshots while the run continues; the one-off
watcher fetches and checksum-verifies the final logs archive on completion.

FOLLOW-UP: BUFF/LeCo measurements in this run are provisional and will be
superseded by ../20261104-buff-leco-native/results-all.txt, which replaces both
codecs with their verified native-access integrations. Final retrieval now
waits for that replacement run.
