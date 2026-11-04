import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path.home()/'gef-benchmark-runs'
RUN=ROOT/'20261103-float32-warnings'
OLD=ROOT/'20261103-recent'
def require(ok,message):
    if not ok: raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    require((RUN/'build.exit').read_text().strip()=='0','Build/tests failed')
    require((RUN/'runner.exit').read_text().strip()=='0','Runner did not finish successfully')
    config=json.loads((RUN/'config.json').read_text())
    manifest=json.loads((RUN/'datasets.json').read_text())
    status=json.loads((RUN/'status.json').read_text())
    summary=json.loads((RUN/'summary.json').read_text())
    require(set(config['codecs'])=={'fauna','macaque'},'Wrong retry scope')
    require(sha(RUN/'LosslessBenchmark')==config['binary_sha256'],'Binary changed')
    require((RUN/'build-tests.txt').read_text().count('Test Passed.')==9,'Missing test passes')
    require(sha(RUN/'source.patch')==(RUN/'source-patch.sha256').read_text().split()[0],'Source patch changed')
    for d in manifest:
        require(sha(Path(d['path']))==d['sha256'],'Dataset changed: '+d['dataset'])
    expected={(d['dataset'],c) for d in manifest for c in ['fauna','macaque']}
    require(len(status)==32 and {(x['dataset'],x['codec']) for x in status}==expected,'Wrong case coverage')
    require(all(x['status'] in ['OK','WARNING'] and x['returncode']==0 for x in status),'Unfinished/failed case')
    require(summary['total']==32 and summary['counts']['FAILED']==0 and summary['counts']['UNSUPPORTED']==0,'Incomplete summary')
    with (RUN/'results.txt').open() as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    require(len(rows)==32 and {(x['dataset'],x['compressor']) for x in rows}==expected,'Missing metrics')
    by_name={d['dataset']:d for d in manifest}
    by_status={(s['dataset'],s['codec']):s for s in status}
    for row in rows:
        n=by_name[row['dataset']]['num_values']; key=(row['dataset'],row['compressor'])
        s=by_status[key]
        require(int(row['num_values'])==n and int(row['uncompressed_bits'])==64*n,'Incomplete dataset measured')
        require(row['input_conversion']=='float32','Missing float32 flag')
        changed=int(row['input_changed_values']); mismatches=int(row['roundtrip_mismatches'])
        require(0<changed<=n and 0<=mismatches<=n,'Invalid diagnostic counts')
        require((row['validation_status']=='passed')==(mismatches==0),'Wrong validation flag')
        require(s['status']=='WARNING' and s['input_changed_values']==changed and s['roundtrip_mismatches']==mismatches,'Wrong ledger diagnostics')
        log=Path(s['log']).read_text()
        require('Warning:' in log and 'Original float64 bits are not preserved' in log,'Missing explicit conversion warning')
        require('error:' not in log,'Hard error reported')
        with (RUN/'cases'/(key[0]+'__'+key[1]+'.metrics.txt')).open() as f:
            original=list(csv.DictReader(f))
        require(len(original)==1 and all(row[k]==v for k,v in original[0].items()),'Aggregate differs from case metrics')
        for field in ['compressed_bits','compression_ratio','compression_throughput_mbs','decompression_throughput_mbs','random_access_ns','random_access_mbs']:
            require(math.isfinite(float(row[field])) and float(row[field])>0,'Invalid metric '+field)
        for r in config['ranges']:
            value=row['range_%d_mbs'%r]
            require(bool(value)==(r<n),'Wrong range coverage')
            if value:require(math.isfinite(float(value)) and float(value)>0,'Invalid range metric')
    with (OLD/'results.txt').open() as f: old=list(csv.DictReader(f))
    old_status=json.loads((OLD/'status.json').read_text())
    require(len(old)==112 and all(x['compressor'] not in ['fauna','macaque'] for x in old),'Wrong original metrics set')
    require({x['compressor'] for x in old}=={'b_star_gef_approximate','dexor','elfstar','selfstar','flea','pcodec','pcodec_i64'},'Unrequested original codec')
    require({x['dataset'] for x in old}==set(by_name),'Wrong original dataset set')
    old_ok={(x['dataset'],x['codec']) for x in old_status if x['status']=='OK'}
    require({(x['dataset'],x['compressor']) for x in old}==old_ok,'Original audit coverage mismatch')
    for row in old:
        with (OLD/'cases'/(row['dataset']+'__'+row['compressor']+'.metrics.txt')).open() as f:
            original=list(csv.DictReader(f))
        require(len(original)==1 and all(row[k]==v for k,v in original[0].items()),'Original metric values changed')
        row.update(input_conversion='none',input_changed_values='0',roundtrip_mismatches='0',validation_status='passed',source_run=OLD.name)
    for row in rows:row['source_run']=RUN.name
    all_rows=old+rows
    require(len({(r['dataset'],r['compressor']) for r in all_rows})==144,'Duplicate combined metrics')
    with (RUN/'results-all.txt').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields+['source_run']);w.writeheader();w.writerows(all_rows)
    bad=sum(int(x['roundtrip_mismatches'])>0 for x in rows)
    text=('Audit passed: 32 affected cases rerun on all 16 complete datasets.\n'
          '32 metric rows emitted with float32 conversion warnings; no hard failures.\n'
          '%d cases have codec reconstruction mismatches (see validation_status).\n'
          'All input hashes, executable hash, source patch, case logs and CSV metadata verified.\n'
          'All nine tests passed, including warning continuation and malformed-length rejection.\n'
          'results-all.txt contains 144 rows: 112 original valid rows plus 32 rerun rows.\n'
          'Original rows retain their measured values; diagnostic fields are populated from their completed audit.\n'
          'Float32 rows measure converted input and do not claim preservation of original float64 values.\n')%bad
    (RUN/'audit.txt').write_text(text);print(text)
if __name__=='__main__':main()
