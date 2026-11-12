#!/usr/local/bin/python3.7
import csv,hashlib,json,sys
from pathlib import Path
R=Path.home()/'gef-benchmark-runs/20261104-all-strict-input'
E=Path.home()/'gef-benchmark-runs/20261104-elf-strict-input'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
config=json.loads((R/'config.json').read_text());ec=json.loads((E/'config.json').read_text())
manifest=json.loads((R/'datasets.json').read_text());em=json.loads((E/'datasets.json').read_text())
assert {(x['dataset'],x['num_values'],x['sha256']) for x in manifest}=={(x['dataset'],x['num_values'],x['sha256']) for x in em}
assert len(config['codecs'])==30 and len(ec['codecs'])==2 and not set(config['codecs'])&set(ec['codecs'])
expected={(d['dataset'],c) for d in manifest for c in config['codecs']+ec['codecs']}
assert len(expected)==512
combined=[];issues=[]
for root,cfg in [(E,ec),(R,config)]:
    assert sha(root/'LosslessBenchmark')==cfg['binary_sha256']
    assert (root/'tests.exit').read_text().strip()=='0'
    statuses=json.loads((root/'status.json').read_text())
    indexed={(x['dataset'],x['codec']):x for x in statuses};assert len(indexed)==len(statuses)
    issues.extend(dict(source_run=root.name,**x) for x in statuses if x['status'] not in ['OK','WARNING'])
    for row in read(root/'results.txt'):
        key=(row['dataset'],row['compressor'].lower())
        s=indexed[key];assert s['status'] in ['OK','WARNING'] and s['returncode']==0
        raw=read(root/'cases'/(key[0]+'__'+key[1]+'.metrics.txt'))
        assert len(raw)==1 and all(row[k]==v for k,v in raw[0].items())
        assert all(not row[k] for k in row.keys()-raw[0].keys())
        item=dict(row,source_run=root.name,case_status=s['status'],input_preprocessing='strict scaled-int64 division')
        item['compressor']=key[1]
        if key[1] in ('fauna','pcodec','pcodec_i64'):
            item['implementation_note']='Existing custom native-access adapter and large-page settings retained; access-method review remains separate.'
        elif key[1] in ('elfstar','selfstar'):
            item['implementation_note']='Existing C++ port of original Java, with counted block framing; corrected input only.'
        else:
            item['implementation_note']='Existing adapter/settings; preserve reported validation limitations, including float32 conversion and legacy not_checked.'
        combined.append(item)
keys={(x['dataset'],x['compressor']) for x in combined};assert len(keys)==len(combined)
missing=sorted(expected-keys)
(R/'missing-cases.json').write_text(json.dumps(missing,indent=2)+'\n')
(R/'failed-cases.json').write_text(json.dumps(issues,indent=2)+'\n')
with (R/'results-all.txt').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
summary=dict(rows=len(combined),expected=512,missing=missing,failed_cases=len(issues),warnings=sum(x['case_status']=='WARNING' for x in combined),source_runs=[E.name,R.name])
(R/'combined-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=[('COMPLETE' if not missing and not issues else 'INCOMPLETE')+': %d/512 corrected-input measurements.'%len(combined),
       'PASS: all 16 dataset counts and SHA-256 hashes agree across both corrected runs.',
       'PASS: each source binary hash matches its run configuration.',
       'PASS: source-run test suites passed before measurements.',
       'PASS: all combined rows match their individual case CSVs.',
       'No rows from the old reciprocal-conversion runs are included.',
       'Warnings retain metrics; legacy validation and float32 qualifications remain explicit.',
       'Fauna/Pcodec adapter and page-size questions are not resolved by this input-conversion rerun.']
(R/'audit.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
sys.exit(0 if not missing and not issues else 1)
