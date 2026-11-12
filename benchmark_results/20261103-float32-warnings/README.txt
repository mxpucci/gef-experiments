Fauna/Macaque warning-and-continue rerun, 2026-11-03

Remote: ~/gef-benchmark-runs/20261103-float32-warnings
Tmux: gef-float32-warnings
Follow progress: tail -f ~/gef-benchmark-runs/20261103-float32-warnings/output.txt

This reruns all 16 full datasets for Fauna and Macaque, retaining the same CPU2,
single-thread setting, 1000-value blocks and point/range workloads as the original run.
The original 112 successful results for other codecs need not be rerun.

Float32 conversion is explicit benchmark preprocessing, included in compression
time. Changes produce a warning, not an exception. Round-trip validation uses
the actual float32 input expanded back to the benchmark double representation.
Metrics retain the existing 8-byte logical-value throughput/ratio denominator.
These are not lossless results for the original float64 values. The original
decimal precision is also lost on BP, BT, BW, LAT and LON (see prior input audit).

New CSV diagnostic columns:
input_conversion: float32 for these codecs; none otherwise.
input_changed_values: count of original values changed by conversion.
roundtrip_mismatches: decoded values differing from the actual encoded input.
validation_status: passed or failed for recent codecs; not_checked for legacy
paths not updated to populate the new field. Conversion loss is independent
of codec validation: passed does not undo preprocessing loss.

Value mismatches warn and still emit measured metrics, marked failed. Hard API
errors or malformed lengths remain isolated case failures: we cannot invent
compression/decompression/query metrics when the operation cannot complete.
The driver isolates each case, logs its outcome, and continues the suite.

results.txt: the 32 rerun rows once complete.
results-all.txt: produced by audit.py; combines the original 112 valid rows and
the 32 new rows, with source_run identifying each measurement's provenance.
output.txt: full progress and warning log.
status.json: per-case outcomes, conversion counts and validation status.
source.patch plus source-base-commit.txt: exact source changes used for this run.
build-tests.txt, audit.txt: validation evidence.
The strict standalone Fauna/Macaque double API and authors' Rust algorithms are unchanged.
