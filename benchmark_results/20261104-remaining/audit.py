#!/usr/local/bin/python3.7
import csv,hashlib,json,sys
from pathlib import Path
R=Path.home()/'gef-benchmark-runs/20261104-remaining'
PREV=Path.home()/'gef-benchmark-runs/20261103-native-access'
def read(name):
    with name.open() as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
config=json.loads((R/'config.json').read_text());manifest=json.loads((R/'datasets.json').read_text())
status=json.loads((R/'status.json').read_text());measured=read(R/'results.txt')
assert config['binary_sha256']==sha(R/'LosslessBenchmark')
assert '100% tests passed, 0 tests failed out of 10' in (R/'tests.txt').read_text()
previous_manifest=json.loads((PREV/'datasets.json').read_text())
assert {(x['dataset'],x['num_values'],x['sha256']) for x in manifest}=={(x['dataset'],x['num_values'],x['sha256']) for x in previous_manifest}
expected={(d['dataset'],c) for d in manifest for c in config['codecs']}
status_by_key={(x['dataset'],x['codec']):x for x in status}
assert len(status_by_key)==len(status)
produced={(x['dataset'],x['compressor'].lower()):x for x in measured}
assert len(produced)==len(measured)
for key,row in produced.items():
    case=status_by_key[key]
    assert case['status'] in ['OK','WARNING'] and case['returncode']==0
    raw=read(R/'cases'/(key[0]+'__'+key[1]+'.metrics.txt'))
    assert len(raw)==1 and all(row[k]==v for k,v in raw[0].items())
missing=sorted(expected-set(produced));failed=[x for x in status if x['status'] not in ['OK','WARNING']]
(R/'missing-cases.json').write_text(json.dumps(missing,indent=2)+'\n')
(R/'failed-cases.json').write_text(json.dumps(failed,indent=2)+'\n')
previous=read(PREV/'results-all.txt');assert len(previous)==144
combined=[]
for row in previous:
    x=dict(row);x['case_status']='PREVIOUS'
    x['implementation_note']='Custom native-access adapter; not out-of-the-box point-query implementation' if row['compressor'] in ['fauna','pcodec','pcodec_i64'] else 'Retained preceding run; see source_run and its provenance'
    combined.append(x)
for row in measured:
    x=dict(row);x['source_run']=R.name
    case=status_by_key[(row['dataset'],row['compressor'].lower())]
    x['case_status']=case['status']
    x['implementation_note']='Existing baseline adapter; legacy validation uses existing checks (BUFF uses a tolerance). See case logs.'
    combined.append(x)
assert len({(x['dataset'],x['compressor'].lower()) for x in combined})==len(combined)
fields=list(measured[0] if measured else previous[0])
fields=[f for f in fields if f!='source_run']+['source_run','case_status','implementation_note']
with (R/'results-all.txt').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(combined)
complete=set(status_by_key)==expected and not any(x['status']=='RUNNING' for x in status)
lines=[('COMPLETE' if complete else 'INCOMPLETE')+': %d/%d cases attempted; %d measured rows; %d missing measurements.'%(len(status),len(expected),len(measured),len(missing)),
       'PASS: all 16 dataset counts and SHA-256 hashes match the previous runs.',
       'PASS: the measured binary hash matches the saved configuration.',
       'PASS: 10/10 existing Linux tests passed before the run.',
       'PASS: every aggregated row matches its individual case CSV.',
       '%d combined rows (%d new, 144 retained); expected 512 if every new case emits metrics.'%(len(combined),len(measured)),
       'WARNING cases retain measured metrics; hard failures are isolated in failed-cases.json.',
       'Legacy validation_status=not_checked is preserved; absence of a warning is not a new bitwise-losslessness guarantee.',
       'Fauna/Pcodec rows retained from the prior run are explicitly qualified as custom native-access adapters.']
(R/'audit.txt').write_text('\n'.join(lines)+'\n')
(R/'binary.sha256').write_text(sha(R/'LosslessBenchmark')+'  LosslessBenchmark\n')
print('\n'.join(lines))
sys.exit(0 if complete and not missing else 1)
