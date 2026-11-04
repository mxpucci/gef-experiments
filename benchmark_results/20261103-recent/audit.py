"""Independent completion audit for the full requested run."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

RUN = Path.home()/'gef-benchmark-runs/20261103-recent'
CODECS = {'b_star_gef_approximate','dexor','elfstar','selfstar','pcodec',
          'pcodec_i64','flea','fauna','macaque'}
NAMES = {'AP','BM','BP','BT','BW','CT','DP','DU','ECG','GE','IT','LAT','LON','UK','US','WD'}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def main():
    require((RUN/'preparation.exit').read_text().strip()=='0','Dataset preparation did not succeed')
    require((RUN/'runner.exit').read_text().strip()=='0','Runner did not complete successfully')
    config = json.loads((RUN/'config.json').read_text())
    datasets = json.loads((RUN/'datasets.json').read_text())
    status = json.loads((RUN/'status.json').read_text())
    summary = json.loads((RUN/'summary.json').read_text())
    require(set(config['codecs'])==CODECS,'Wrong requested codec set')
    require(len(datasets)==16 and {d['dataset'] for d in datasets}==NAMES,'Wrong dataset inventory')
    require(sha256(Path(config['binary']))==config['binary_sha256'],'Benchmark binary changed')
    for d in datasets:
        p = Path(d['path'])
        require(p.stat().st_size==d['bytes'] and sha256(p)==d['sha256'],'Dataset changed: '+d['dataset'])
    expected = {(d,c) for d in NAMES for c in CODECS}
    observed = [(x['dataset'],x['codec']) for x in status]
    require(len(observed)==144 and set(observed)==expected,'Incomplete or duplicate case coverage')
    counts = collections.Counter(x['status'] for x in status)
    require(set(counts)<= {'OK','UNSUPPORTED'},'Unresolved failed/running cases')
    require(summary['total']==144,'Incorrect completion summary')
    for key in ['OK','UNSUPPORTED','FAILED']:
        require(summary['counts'][key]==counts[key],'Summary count mismatch')
    with (RUN/'results.txt').open() as f:
        rows = list(csv.DictReader(f))
    actual = [(x['dataset'],x['compressor']) for x in rows]
    successful = {(x['dataset'],x['codec']) for x in status if x['status']=='OK'}
    require(len(actual)==len(successful) and set(actual)==successful,'Combined result rows mismatch successful cases')
    combined = {(x['dataset'],x['compressor']):x for x in rows}
    by_name = {d['dataset']:d for d in datasets}
    report = ['Full benchmark audit passed.',
              'All 16 complete datasets and all 9 requested entries were attempted (144 cases).',
              'Successful metrics: %d; unsupported inputs: %d; failures: 0.' % (counts['OK'],counts['UNSUPPORTED']),
              'Binary hash and all dataset hashes verified after completion.',
              '', 'Per-codec outcomes:']
    for codec in sorted(CODECS):
        statuses = collections.Counter(x['status'] for x in status if x['codec']==codec)
        report.append('%s: %d OK, %d UNSUPPORTED' % (codec,statuses['OK'],statuses['UNSUPPORTED']))
    for x in status:
        require(x['finished']>=x['started'],'Invalid case timestamps')
        require(x['command'][0:3]==['taskset','-c','2'],'Wrong CPU binding')
        require(x['command'][-1]==by_name[x['dataset']]['path'],'Wrong input path')
        require(x['command'][x['command'].index('-c',3)+1]==x['codec'],'Wrong codec invocation')
        log = Path(x['log']).read_text()
        if x['status']=='UNSUPPORTED':
            require(x['returncode']==1,'Unsupported case crashed')
            allowed = (x['codec'] in ['fauna','macaque'] and
                       'upstream supports f32 only; input is not exactly representable' in log)
            allowed |= (x['codec']=='flea' and
                        'block sum(abs(values)) exceeds safe imported implementation limit' in log)
            require(allowed,'Unrecognized unsupported input reason')
            continue
        require(x['returncode']==0,'Successful case has nonzero exit')
        require('decompression error' not in log and 'error:' not in log and 'Warning:' not in log,
                'Diagnostic warning/error in successful case')
        metric_path = RUN/'cases'/(x['dataset']+'__'+x['codec']+'.metrics.txt')
        with metric_path.open() as f:
            original = list(csv.DictReader(f))
        require(len(original)==1,'Wrong original metrics row count')
        row = combined[(x['dataset'],x['codec'])]
        require(all(row[k]==v for k,v in original[0].items()),'Aggregate differs from original metrics')
        n = by_name[x['dataset']]['num_values']
        require(int(row['num_values'])==n and int(row['uncompressed_bits'])==n*64,'Incomplete data measured')
        for key,value in row.items():
            if key not in ['dataset','compressor'] and value:
                require(math.isfinite(float(value)) and float(value)>0,'Invalid metric '+key)
        for r in config['ranges']:
            require(bool(row['range_%d_mbs'%r])==(r<n),'Missing/unexpected range metric')
    report += ['', 'No unrequested compressors were run. GEF uses its native access methods.',
               'Pcodec results use independent 1000-value standalone chunks; native shared-metadata paging was not used.',
               'See README.txt and repository docs/recent-compressors.md for representation and timing details.']
    text = '\n'.join(report)+'\n'
    (RUN/'audit.txt').write_text(text)
    print(text)

if __name__=='__main__':
    main()
