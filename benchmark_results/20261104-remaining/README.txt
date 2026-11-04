LOCAL SNAPSHOT: run launched; full results are still being produced on the server.

Remaining compressor benchmark — 2026-11-04

Run: ~/gef-benchmark-runs/20261104-remaining on testPF@131.114.142.6
Tmux session: gef-remaining. output.txt/status.json show progress.

23 entries x 16 complete datasets = 368 isolated cases:
dac, rle_gef, u_gef_approximate, u_gef_optimal, b_gef_approximate,
b_gef_optimal, b_star_gef_optimal, gorilla, chimp, chimp128, tsxor,
elf, camel, falcon, alp, buff, leco, lz4, zstd, snappy, xz, brotli, neats.
Already-completed newer compressors and b_star_gef_approximate are not rerun.

One process per case, sequential, CPU 2; no data sampling or case timeout.
Same existing query counts, ranges, seeds and codec settings as the harness.
External -b 1000; GEF native uniform partition size 32000; ALP/Falcon use their
existing internal settings. No new access algorithms introduced for this run.
Legacy native-access codecs retain their pre-existing adapters/settings.

The full target is built with the same GCC 13.4 toolchain, native CPU flags,
OpenMP disabled, and all optional baseline codecs enabled. Squash 0.7.0 uses
its original plugins. LeCo uses the existing CI compatibility fixes (duplicate
standard-library type traits and deprecated register keyword) in generated
header copies; source submodules are unchanged. It is compiled with the same
GCC/libstdc++ ABI using the recorded leco-cxx wrapper. Exact dependency package
records, compatibility patches, source snapshots and build commands are saved.
Additional dependencies and datasets reside in /dev/shm and require recreation
after a server reboot. The build scripts and pinned manifests are retained.

Warnings retain metrics. Crashes/hard errors fail only the individual case.
Legacy validation_status remains not_checked; the existing checks are not
upgraded to bitwise validation. BUFF's existing check uses a tolerance.
A validation warning is recorded in status.json, not silently counted as valid.

On completion: audit.py verifies dataset hashes, case coverage, binary identity
and case CSV agreement; it writes audit.txt, missing-cases.json and
failed-cases.json. results-all.txt combines the new measurements with 144
previous measurements, with source_run/case_status/implementation_note fields.
The earlier Fauna/Pcodec rows are explicitly qualified as custom native-access
adapters, not out-of-the-box point-query implementations. They are not revised
by this run. Expected combined rows: 512 if all new cases emit metrics.

The final logs archive is ~/gef-benchmark-runs/20261104-remaining.tar.gz;
archive.sha256 records its digest. The benchmark binary is retained on the
server and excluded from the downloadable logs archive.
