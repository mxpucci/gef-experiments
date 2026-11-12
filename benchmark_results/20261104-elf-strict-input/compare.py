#!/usr/bin/env python3
"""Compare corrected Elf*/SElf* measurements with their preceding run."""
import csv
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
previous = root.parent / '20261103-native-access/results-all.txt'
def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))
old = {(r['dataset'], r['compressor']): r for r in read(previous)}
new = read(root / 'results.txt')
assert len({(r['dataset'], r['compressor']) for r in new}) == len(new)
rows = []
for r in new:
    assert r['compressor'] in ('elfstar', 'selfstar')
    o = old[(r['dataset'], r['compressor'])]
    assert r['num_values'] == o['num_values']
    n = int(r['num_values'])
    before, after = int(o['compressed_bits']), int(r['compressed_bits'])
    x = dict(dataset=r['dataset'], compressor=r['compressor'], num_values=n,
             old_bits_per_value=before/n, new_bits_per_value=after/n,
             old_compressed_percent=100*before/(64*n), new_compressed_percent=100*after/(64*n),
             compressed_size_reduction_percent=100*(1-after/before),
             validation_status=r['validation_status'], roundtrip_mismatches=r['roundtrip_mismatches'])
    for metric in ('compression_throughput_mbs', 'decompression_throughput_mbs', 'random_access_mbs'):
        x['old_'+metric] = float(o[metric]); x['new_'+metric] = float(r[metric])
    rows.append(x)
if rows:
    with (root/'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
summary = {}
for codec in ('elfstar', 'selfstar'):
    part = [r for r in rows if r['compressor'] == codec]
    if not part: continue
    summary[codec] = dict(datasets=len(part),
                         mean_old_bits_per_value=mean(r['old_bits_per_value'] for r in part),
                         mean_new_bits_per_value=mean(r['new_bits_per_value'] for r in part),
                         mean_old_compressed_percent=mean(r['old_compressed_percent'] for r in part),
                         mean_new_compressed_percent=mean(r['new_compressed_percent'] for r in part),
                         mean_old_compression_mib_s=mean(r['old_compression_throughput_mbs'] for r in part),
                         mean_new_compression_mib_s=mean(r['new_compression_throughput_mbs'] for r in part),
                         mean_old_decompression_mib_s=mean(r['old_decompression_throughput_mbs'] for r in part),
                         mean_new_decompression_mib_s=mean(r['new_decompression_throughput_mbs'] for r in part),
                         mean_old_point_mib_s=mean(r['old_random_access_mbs'] for r in part),
                         mean_new_point_mib_s=mean(r['new_random_access_mbs'] for r in part))
(root/'comparison-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines = ['Elf*/SElf*: strict decimal-input conversion comparison',
         'Completed measurements: %d/32. Old results retain reciprocal-conversion issue.'%len(rows),
         'All sizes include the existing headers, checkpoints and block directory.',
         'Bits/value; lower is better. Averages give equal weight to datasets.',
         '', 'Dataset     Elf* old -> new       SElf* old -> new']
for dataset in sorted({r['dataset'] for r in rows}):
    parts = []
    for codec in ('elfstar', 'selfstar'):
        p = next((r for r in rows if (r['dataset'],r['compressor'])==(dataset,codec)),None)
        parts.append('%7.3f -> %7.3f'%(p['old_bits_per_value'],p['new_bits_per_value']) if p else 'pending')
    lines.append('%-8s  %-23s %s'%(dataset,*parts))
lines += ['', json.dumps(summary,indent=2),
          'Only Elf*/SElf* rerun. Other converted-double codec results remain provisional.',
          'Implementations remain the existing C++ ports; this change corrects shared input preparation.']
(root/'comparison.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
