GEF recent-competitor benchmark run, 2026-11-03

Host: testPF@131.114.142.6 (dl580-gpu3.unipi.it)
Persistent run directory: ~/gef-benchmark-runs/20261103-recent
Tmux session: gef-recent
Attach: tmux attach -t gef-recent
Follow the text log: tail -f ~/gef-benchmark-runs/20261103-recent/output.txt

output.txt: combined progress and per-case diagnostic output.
results.txt: validated successful metrics, comma-separated text with one header.
status.json: every attempted case, exit status, timestamps, diagnostics.
summary.json and runner.exit: written at completion.
cases/: original per-case logs and raw metrics; failed cases are retained.
datasets.json: complete dataset inventory, counts, decimal scales and SHA-256.
config.json and environment.txt: binary/source identity, hardware and settings.
prepare.sh, launch.sh and run.py: exact run scripts.

All 16 complete datasets in the original split archive are used, without sampling.
Dataset staging: /dev/shm/testPF-gef-20261103 (temporary RAM filesystem).
Text outputs and the executable snapshot persist in the home directory.
Do not remove staging files while the benchmark is running.

Scope: b_star_gef_approximate, dexor, elfstar, selfstar, pcodec,
pcodec_i64, flea, fauna, macaque. Kangaroo and EAS are deferred.
All other baseline compressors are excluded.

One process per case, sequential, pinned to logical CPU 2; one thread.
Native release build, GEF SIMD enabled, OpenMP disabled.
New-codec block size: 1000 values. GEF uniform partition size: 32000.
Range sizes: 10, 100, 1000, 10000, 100000 (only sizes below dataset length).
Existing harness query counts: GEF 1M points and 10K per range;
new codecs 100K points and 1K per range. Rates use actual counts.
Pcodec uses default level 8. The 1000-value access experiment does not measure
its best compression ratio, for which upstream recommends larger blocks.
Fauna/Macaque accept only exact float32 inputs through the double adapter.
FLEA uses raw scaled integers and its documented safe magnitude bounds.
Unsupported inputs are recorded explicitly, without rounding or codec substitution.
Integer codecs use raw int64 data; floating codecs use the existing harness's
float64 conversion. See repository docs/recent-compressors.md for accounting.

Successful metrics require the expected dataset name and full value count,
finite positive metrics, successful exit, and no reported reconstruction error.
The compressor harness validates full reconstruction before returning results.
Missing metrics or a crash are FAILED, not silently treated as unsupported.
No wall-clock timeout is used to discard slow full-dataset runs.
