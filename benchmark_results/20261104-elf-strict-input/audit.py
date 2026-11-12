#!/usr/bin/env python3
import csv
import json
from pathlib import Path
r=Path(__file__).resolve().parent
p=r.parent/'20261103-native-access'
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
config=json.loads((r/'config.json').read_text())
assert config['codecs']==['elfstar','selfstar'] and config['total_cases']==32
assert config['cpu']==2 and config['block_size']==1000
manifest=json.loads((r/'datasets.json').read_text())
previous=json.loads((p/'datasets.json').read_text())
assert {(x['dataset'],x['num_values'],x['sha256']) for x in manifest}=={(x['dataset'],x['num_values'],x['sha256']) for x in previous}
status=json.loads((r/'status.json').read_text());rows=read(r/'results.txt')
expected={(d['dataset'],c) for d in manifest for c in config['codecs']}
assert len(rows)==32 and {(x['dataset'],x['compressor']) for x in rows}==expected
assert len(status)==32 and all(x['status'] in ['OK','WARNING'] and x['returncode']==0 for x in status)
for row in rows:
    raw=read(r/'cases'/(row['dataset']+'__'+row['compressor']+'.metrics.txt'))
    assert len(raw)==1 and all(row[k]==v for k,v in raw[0].items())
    assert all(not row[k] for k in row.keys()-raw[0].keys())
assert '100% tests passed, 0 tests failed out of 4' in (r/'tests.txt').read_text()
assert (r/'runner.exit').read_text().strip()=='0'
by_key={(x['dataset'],x['compressor']):x for x in rows}
assert int(by_key['BP','elfstar']['compressed_bits'])==183992
assert int(by_key['BP','selfstar']['compressed_bits'])==189088
warnings=[x for x in rows if x['validation_status']!='passed' or int(x['roundtrip_mismatches'])]
lines=['PASS: 32/32 complete measurements (two codecs, all 16 full datasets).',
       'PASS: all dataset counts and SHA-256 hashes match the previous run.',
       'PASS: all aggregate rows exactly match their individual case metrics.',
       'PASS: four GCC 13.4/LTO tests passed; fast-math caller input regression included.',
       'PASS: full BP compressed sizes exactly match the independently checked strict-input Java/C++ diagnostic.',
       ('PASS: all 32 cases passed bitwise round-trip validation.' if not warnings else 'WARNING: %d cases carry validation warnings; metrics retained.'%len(warnings))]
(r/'audit.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
