#!/usr/bin/env python3
"""End-to-end CLI/CSV checks; requires the full set of new codec build options."""
import csv
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())

def dataset(path, values, decimals=0):
    path.write_bytes(struct.pack("<QQ", len(values), decimals) +
                     struct.pack("<" + "q" * len(values), *values))

def run(path, names, success=True):
    p = subprocess.run([binary, "-b", "16", "-r", "3,35", "-c", ",".join(names), str(path)],
                       capture_output=True, text=True)
    if (p.returncode == 0) != success:
        raise RuntimeError("unexpected status: " + p.stderr)
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    if success:
        if {row["compressor"] for row in rows} != set(names):
            raise RuntimeError("missing/duplicate CSV compressor: " + p.stdout + p.stderr)
        if len(rows) != len(names):
            raise RuntimeError("incorrect number of rows")
        for row in rows:
            if int(row["num_values"]) != 97 or int(row["compressed_bits"]) <= 0:
                raise RuntimeError("invalid row: " + repr(row))
    elif rows:
        raise RuntimeError("failed input emitted successful metrics")
    return p

with tempfile.TemporaryDirectory(prefix="gef-codec-smoke-") as directory:
    root = Path(directory)
    exact = root / "quarters.bin"
    dataset(exact, [(i * 13 % 97 - 48) * 25 for i in range(97)], 2)
    names = ["fauna", "macaque", "dexor", "pcodec", "pcodec_i64", "elfstar", "selfstar", "flea"]
    run(exact, names)
    # The integer route must retain all 64 bits without passing through double.
    integers = root / "integers.bin"
    dataset(integers, [-(1 << 63) + i for i in range(97)])
    run(integers, ["pcodec_i64"])
    inexact = root / "hundredths.bin"
    dataset(inexact, [i + 1 for i in range(97)], 2)
    for name in ["fauna", "macaque"]:
        p = run(inexact, [name], success=False)
        if "not exactly representable" not in p.stderr:
            raise RuntimeError("missing eligibility explanation")
    run(exact, ["unknown_codec"], success=False)
print("CLI smoke: all eight entries, exact int64 route, eligibility errors and CSV passed")
