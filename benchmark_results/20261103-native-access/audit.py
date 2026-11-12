#!/usr/local/bin/python3.7
import csv, hashlib, json, math, statistics
from pathlib import Path
R=Path.home()/'gef-benchmark-runs/20261103-native-access'
OLD=Path.home()/'gef-benchmark-runs/20261103-float32-warnings'
BASE=Path.home()/'gef-benchmark-runs/20261103-recent'
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,items):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
assert (R/'runner.exit').read_text().strip()=='0'
assert (R/'build.exit').read_text().strip()=='0'
assert '100% tests passed, 0 tests failed out of 10' in (R/'tests.txt').read_text()
status=json.loads((R/'status.json').read_text());conf=json.loads((R/'config.json').read_text())
assert len(status)==48 and all(x['status'] in ['OK','WARNING'] and x['returncode']==0 for x in status)
assert conf['binary_sha256']==sha(R/'LosslessBenchmark')
manifest=json.loads((R/'datasets.json').read_text());previous=json.loads((BASE/'datasets.json').read_text())
assert {(x['dataset'],x['num_values'],x['sha256']) for x in manifest}=={(x['dataset'],x['num_values'],x['sha256']) for x in previous}
new=rows(R/'results.txt');old=rows(OLD/'results-all.txt');replacements={'fauna','pcodec','pcodec_i64'}
assert len(new)==48 and len(old)==144
key=lambda x:(x['dataset'],x['compressor'])
assert len({key(x) for x in new})==48
assert {key(x) for x in new}=={key(x) for x in old if x['compressor'] in replacements}
counts={x['dataset']:x['num_values'] for x in manifest}
for row in new:
    assert int(row['num_values'])==counts[row['dataset']]
    assert int(row['uncompressed_bits'])==counts[row['dataset']]*64
    assert row['validation_status']=='passed' and int(row['roundtrip_mismatches'])==0
    fauna=row['compressor']=='fauna'
    assert row['input_conversion']==('float32' if fauna else 'none')
    assert row['access_layout']==('native_segments' if fauna else 'native_pages')
    assert int(row['compression_unit_values'])==(65536 if fauna else 262144)
    assert int(row['native_page_values'])==(0 if fauna else 262144)
    if not fauna:assert int(row['input_changed_values'])==0
    case=rows(R/'cases'/(row['dataset']+'__'+row['compressor']+'.metrics.txt'))
    assert len(case)==1 and all(row[k]==v for k,v in case[0].items())
lookup={key(x):x for x in new};allrows=[]
for item in old:
    if item['compressor'] in replacements:
        row=dict(lookup[key(item)]);row['source_run']=R.name
    else:
        row=dict(item)
        row['access_layout']='unspecified' if item['compressor']=='b_star_gef_approximate' else 'external_blocks'
        row['compression_unit_values']='0' if item['compressor']=='b_star_gef_approximate' else '1000'
        row['native_page_values']='0'
    allrows.append({k:row[k] for k in list(new[0])+['source_run']})
assert len(allrows)==144 and len({key(x) for x in allrows})==144
write('results-all.txt',allrows)
oldmap={key(x):x for x in old};comparison=[]
for x in new:
    y=oldmap[key(x)]
    comparison.append(dict(dataset=x['dataset'],compressor=x['compressor'],
        old_compressed_bits=y['compressed_bits'],native_compressed_bits=x['compressed_bits'],
        native_size_over_old=float(x['compressed_bits'])/float(y['compressed_bits']),
        old_random_access_ns=y['random_access_ns'],native_random_access_ns=x['random_access_ns'],
        native_point_time_over_old=float(x['random_access_ns'])/float(y['random_access_ns']),
        old_decompression_mbs=y['decompression_throughput_mbs'],native_decompression_mbs=x['decompression_throughput_mbs']))
write('comparison.txt',comparison)
lines=['PASS: 48/48 native-layout cases completed; 10/10 Linux tests passed.',
       'PASS: all full-input reconstructions passed against the declared input representation.',
       'PASS: 16 dataset counts and SHA-256 hashes match the earlier complete run.',
       'PASS: benchmark binary hash matches the run configuration.',
       'PASS: 144 unique combined rows: 48 replaced, 96 earlier measurements preserved.',
       'Fauna still narrows to float32; warnings do not imply original-f64 losslessness.',
       'GEF layout fields are unspecified/zero because its original CSV predates these fields.',
       'Size ratios aggregate encoded bytes over all 16 datasets; point-time ratios are geometric means.']
for codec in sorted(replacements):
    a=[x for x in new if x['compressor']==codec];b=[x for x in old if x['compressor']==codec]
    ratio=sum(int(x['compressed_bits']) for x in a)/sum(int(x['compressed_bits']) for x in b)
    points=[float(x['native_point_time_over_old']) for x in comparison if x['compressor']==codec]
    geo=math.exp(sum(math.log(x) for x in points)/len(points))
    mean_old=statistics.mean(float(x['compressed_bits'])/float(x['uncompressed_bits']) for x in b)
    mean_new=statistics.mean(float(x['compressed_bits'])/float(x['uncompressed_bits']) for x in a)
    lines.append('%s: aggregate native/old bytes %.6f; geometric-mean native/old point time %.6f; equal-dataset mean compressed/raw %.6f -> %.6f'%(codec,ratio,geo,mean_old,mean_new))
(R/'audit.txt').write_text('\n'.join(lines)+'\n')
(R/'binary.sha256').write_text(sha(R/'LosslessBenchmark')+'  LosslessBenchmark\n')
(R/'source-patch.sha256').write_text(sha(R/'source.patch')+'  source.patch\n')
(R/'source-base-commit.txt').write_text(conf['source_commit']+'\n')
(R/'README.txt').write_text('''Corrected native-access benchmark, 2026-11-03

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
''')
print('\n'.join(lines))
