Corrected native-access benchmark, 2026-11-03

results.txt: 48 fresh measurements (Fauna, Pcodec f64, Pcodec i64, 16 datasets).
results-all.txt: these 48 plus 96 unchanged measurements from prior runs.
comparison.txt: per-dataset old external-block versus native measurements.
audit.txt: validation and numerical summary. source-review.txt: source findings.
source.patch: complete local changes against source-base-commit.txt, including
prior warning handling and native access. source-files.tar.gz contains exact
modified/new files. before-native files back up the preceding remote state.

External -b 1000 is passed but does not control these three entries.
Fauna uses native fitting batches of 65536 and indexes native segments.
Pcodec uses native chunks/pages capped at 262144 (original pcopage defaults).
Points scan prefixes within Gorilla segments/Pcodec pages; Fauna PMC/Swing
models evaluate only requested positions. All stored index and metadata bytes
count. Per-query native decoder initialization is timed.
Each process is pinned to CPU 2; 100000 points and 1000 queries per valid range.
Full datasets are used. Timings include allocation/FFI/representation costs.
Fauna float32 conversion warnings and changed-value counts remain explicit;
validation passes against converted input, not original float64 bits.
The benchmark binary stays on the server; hashes and detailed logs are retained.
