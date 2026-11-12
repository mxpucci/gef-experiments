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
    p = run(exact, names)
    for row in csv.DictReader(io.StringIO(p.stdout)):
        if row['validation_status'] != 'passed' or int(row['roundtrip_mismatches']) != 0:
            raise RuntimeError('exact inputs did not pass validation')
        if int(row['input_changed_values']) != 0:
            raise RuntimeError('exact inputs were incorrectly reported as changed')
        if row['compressor'] in ['fauna','macaque','pcodec','pcodec_i64'] and (row['access_layout']!='external_blocks' or int(row['compression_unit_values'])!=16):
            raise RuntimeError(row['compressor'] + ' must use external -b blocks')
    # The integer route must retain all 64 bits without passing through double.
    integers = root / "integers.bin"
    dataset(integers, [-(1 << 63) + i for i in range(97)])
    run(integers, ["pcodec_i64"])
    inexact = root / "hundredths.bin"
    dataset(inexact, [i + 1 for i in range(97)], 2)
    # Values that float32 cannot represent (hundredths, and 9 significant
    # digits) must be compressed as unconverted float64 by every codec.
    precision_loss = root / 'precision_loss.bin'
    dataset(precision_loss, [197054018+i for i in range(97)], 4)
    for path, names in [(inexact, ["fauna", "macaque", "dexor"]), (precision_loss, ['fauna','macaque','pcodec'])]:
        p = run(path, names)
        for row in csv.DictReader(io.StringIO(p.stdout)):
            if row['input_conversion'] != 'none' or int(row['input_changed_values']) != 0:
                raise RuntimeError('unexpected input conversion: ' + repr(row))
            if row['validation_status'] != 'passed' or int(row['roundtrip_mismatches']) != 0:
                raise RuntimeError('codec did not preserve its float64 input: ' + repr(row))
        if 'Warning:' in p.stderr:
            raise RuntimeError('unexpected warning: ' + p.stderr)
    run(exact, ["unknown_codec"], success=False)
print("CLI smoke: all eight entries, exact int64 route, unconverted float64 input and CSV passed")
