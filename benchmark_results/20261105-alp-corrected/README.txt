Corrected ALP full-dataset benchmark, 2026-11-05

All 16 full datasets; 1,482,981,623 values total. Every case passed bitwise full
reconstruction and the adapter point/range checks. Production LosslessBenchmarkFull
was built in /home/testPF/gef-experiments and copied to an immutable run binary.
CPU 8, standard suite settings: -c alp -b 1000 -r 10,100,1000,10000,100000.
ALP ignores external -b and uses native 1024-value vectors / 100-vector groups.
All measured fields (compression, decompression, point/range access and size)
are from the corrected run. See launch.sh, run.py, binary.sha256 and complete.json.

The prepool plot keeps its original 14 common datasets and replaces ALP fields
only. The other two ALP rows remain available here. Raw per-case logs/CSVs are
preserved locally under cases/ and on the server (raw cases excluded from Git).
Comparison with the old plotted ALP rows is in comparison.csv.

Canonical source provenance is captured in source-manifest.json. Prior source
files were backed up on the server under source-before before the build.
