#!/usr/local/bin/python3.7
import csv,hashlib,json,sys
from pathlib import Path
R=Path.home()/'gef-benchmark-runs/20261104-buff-leco-native'
B=Path.home()/'gef-benchmark-runs/20261104-all-strict-input'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
cfg=json.loads((R/'config.json').read_text());manifest=json.loads((R/'datasets.json').read_text())
base_manifest=json.loads((B/'datasets.json').read_text())
assert {(x['dataset'],x['num_values'],x['sha256']) for x in manifest}=={(x['dataset'],x['num_values'],x['sha256']) for x in base_manifest}
assert sha(R/'LosslessBenchmark')==cfg['binary_sha256']
assert (R/'build.exit').read_text().strip()=='0'
status=json.loads((R/'status.json').read_text());index={(x['dataset'],x['codec']):x for x in status}
new=read(R/'results.txt');expected={(x['dataset'],c) for x in manifest for c in ['buff','leco']}
produced=set();newrows=[]
for row in new:
    key=(row['dataset'],row['compressor'].lower());assert key not in produced;produced.add(key)
    case=index[key];assert case['status'] in ['OK','WARNING'] and case['returncode']==0
    raw=read(R/'cases'/(key[0]+'__'+key[1]+'.metrics.txt'))
    assert len(raw)==1 and all(row[k]==v for k,v in raw[0].items())
    assert all(not row[k] for k in row.keys()-raw[0].keys())
    if key[1]=='buff':
        assert row['access_layout']=='native_columns' and row['compression_unit_values']==row['num_values']
    else:
        assert row['access_layout']=='native_fixed_partitions_tuned'
        assert int(row['compression_unit_values']) in [200,400,800,1600,3200,6400,10000]
    item=dict(row,source_run=R.name,case_status=case['status'],input_preprocessing='strict scaled-int64 division')
    item['compressor']=key[1]
    item['implementation_note']=('Original scalar Rust method bodies, whole native column, two-expression selective-decoder fix; bounded precision, not binary64 bitwise lossless.' if key[1]=='buff' else 'Original uint64 C++ LeCo-fix, native 1% partition sampler; generic external block setting ignored.')
    newrows.append(item)
old=read(B/'results-all.txt')
combined=[x for x in old if x['compressor'].lower() not in ['buff','leco']]+newrows
keys={(x['dataset'],x['compressor'].lower()) for x in combined};assert len(keys)==len(combined)
all_cfg=json.loads((B/'config.json').read_text())
all_codecs=all_cfg['codecs']+['elfstar','selfstar']
all_expected={(d['dataset'],c) for d in manifest for c in all_codecs}
missing=sorted(all_expected-keys)
failures=[x for x in status if x['status'] not in ['OK','WARNING']]
with (R/'results-all.txt').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
summary=dict(rows=len(combined),expected=512,replacement_rows=len(newrows),replacement_expected=32,missing=missing,failed_cases=len(failures),warnings=sum(x['case_status']=='WARNING' for x in combined),source_runs=sorted({x['source_run'] for x in combined}))
(R/'combined-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(R/'missing-cases.json').write_text(json.dumps(missing,indent=2)+'\n')
(R/'failed-cases.json').write_text(json.dumps(failures,indent=2)+'\n')
lines=[('COMPLETE' if not missing and not failures else 'INCOMPLETE')+': %d/512 measurements; %d/32 native BUFF/LeCo replacements.'%(len(combined),len(newrows)),
       'PASS: dataset hashes/counts match the full strict-input rerun.',
       'PASS: every replacement row matches its individual case metrics.',
       'PASS: BUFF uses one native column; LeCo uses a sampled native partition size.',
       'All old BUFF/LeCo rows removed, including if a replacement measurement is missing.',
       'BUFF retains its bounded-precision qualification and explicit mismatch counts.',
       'Other codec rows retain their original implementation/access qualifications.']
(R/'audit.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
sys.exit(0 if not missing and not failures else 1)
