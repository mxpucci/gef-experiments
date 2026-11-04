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
            raise RuntimeError('exact float32 inputs were incorrectly reported as changed')
        if row['compressor']=='fauna' and (row['access_layout']!='native_segments' or int(row['compression_unit_values'])!=65536):
            raise RuntimeError('Fauna still uses external -b blocking')
        if row['compressor'] in ['pcodec','pcodec_i64'] and (row['access_layout']!='native_pages' or int(row['compression_unit_values'])!=262144):
            raise RuntimeError('Pcodec still uses external -b blocking')
    # External -b must not change native compression layouts or stored size.
    def native_run(block):
        args=[binary,'-b',str(block),'-r','3,35','-c','fauna,pcodec,pcodec_i64',
              '--fauna-batch-size','53','--pcodec-chunk-size','61','--pcodec-page-size','19',str(exact)]
        process=subprocess.run(args,capture_output=True,text=True)
        if process.returncode:raise RuntimeError(process.stderr)
        return {r['compressor']:r for r in csv.DictReader(io.StringIO(process.stdout))}
    small,large=native_run(7),native_run(1000)
    for name in small:
        if small[name]['compressed_bits']!=large[name]['compressed_bits']:
            raise RuntimeError('external block size changed native layout')
        if small[name]['validation_status']!='passed':raise RuntimeError('native query validation failed')
        if name.startswith('pcodec') and small[name]['native_page_values']!='19':raise RuntimeError('native page option ignored')
    # The integer route must retain all 64 bits without passing through double.
    integers = root / "integers.bin"
    dataset(integers, [-(1 << 63) + i for i in range(97)])
    run(integers, ["pcodec_i64"])
    inexact = root / "hundredths.bin"
    dataset(inexact, [i + 1 for i in range(97)], 2)
    p = run(inexact, ["fauna", "macaque", "dexor"])
    for row in csv.DictReader(io.StringIO(p.stdout)):
        if row['compressor'] in ['fauna','macaque']:
            if row['input_conversion'] != 'float32' or int(row['input_changed_values']) == 0:
                raise RuntimeError('missing input conversion flag/count')
            if row['validation_status'] != 'passed' or int(row['roundtrip_mismatches']) != 0:
                raise RuntimeError('codec did not preserve its converted float32 input')
        elif row['input_conversion'] != 'none' or int(row['input_changed_values']) != 0:
            raise RuntimeError('float32 preprocessing affected another codec')
    if p.stderr.count('Warning:') != 2 or 'Original float64 bits are not preserved' not in p.stderr:
        raise RuntimeError('expected one explicit conversion warning per affected codec')
    # Conversion can also change the original fixed-decimal values, not just
    # their binary64 representation. Still emit flagged, validated metrics.
    precision_loss = root / 'precision_loss.bin'
    dataset(precision_loss, [197054018+i for i in range(97)], 4)
    p = run(precision_loss, ['fauna','macaque','pcodec'])
    for row in csv.DictReader(io.StringIO(p.stdout)):
        if row['validation_status'] != 'passed':
            raise RuntimeError('converted-input round trip failed')
        if row['compressor'] in ['fauna','macaque'] and int(row['input_changed_values']) == 0:
            raise RuntimeError('precision loss was not recorded')
    run(exact, ["unknown_codec"], success=False)
print("CLI smoke: all eight entries, exact int64 route, conversion warnings, continued metrics and CSV passed")
