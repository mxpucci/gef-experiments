#!/usr/local/bin/python3.7
"""Run every dataset/codec pair sequentially, retain evidence, validate rows."""
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

RUN = Path.home() / 'gef-benchmark-runs/20261104-remaining'
SOURCE = Path.home() / 'gef-experiments'
DATA = Path('/dev/shm/testPF-gef-20261103/extracted')
CODECS = ['dac','rle_gef','u_gef_approximate','u_gef_optimal','b_gef_approximate','b_gef_optimal','b_star_gef_optimal','gorilla','chimp','chimp128','tsxor','elf','camel','falcon','alp','buff','leco','lz4','zstd','snappy','xz','brotli','neats']
EXPECTED = {'AP':137721453,'BM':17964,'BP':6782,'BT':129669,'BW':129669,
            'CT':2826654,'DP':5413914,'DU':334880,'ECG':225750000,
            'GE':43565657,'IT':477050760,'LAT':24876978,'LON':24876978,
            'UK':59305325,'US':282076178,'WD':198898762}
RANGES = [10,100,1000,10000,100000]
FIELDS = ['compressor','dataset','num_values','uncompressed_bits','compressed_bits',
          'compression_ratio','compression_throughput_mbs','decompression_throughput_mbs',
          'random_access_ns','random_access_mbs'] + ['range_%d_mbs'%r for r in RANGES] + ['input_conversion','input_changed_values','roundtrip_mismatches','validation_status','access_layout','compression_unit_values','native_page_values']

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def log(message):
    print(stamp() + ' ' + message, flush=True)

def save_json(name, value):
    path = RUN / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    os.environ.update(LC_ALL='C', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    (RUN/'runner.pid').write_text(str(os.getpid()) + '\n')
    (RUN/'cases').mkdir(exist_ok=True)
    if (RUN/'status.json').exists():
        raise RuntimeError('Existing run status: refusing to overwrite a previous run')
    log('Validating all datasets; full input, no sampling')
    manifest = []
    for path in sorted(DATA.rglob('*.bin')):
        with path.open('rb') as f:
            count = struct.unpack('<Q', f.read(8))[0]
            size = path.stat().st_size
            if size == count*8+16:
                decimals = struct.unpack('<Q', f.read(8))[0]
            elif size == count*8+8:
                decimals = 0
            else:
                raise RuntimeError('Invalid dataset size: ' + str(path))
        name = path.stem
        if name not in EXPECTED or count != EXPECTED[name]:
            raise RuntimeError('Unexpected dataset/count: %s %d' % (name,count))
        manifest.append(dict(dataset=name, path=str(path), num_values=count,
                             decimals=decimals, bytes=size, sha256=sha256(path)))
        log('Verified %s: %d values, decimals=%d' % (name,count,decimals))
    if len(manifest) != len(EXPECTED) or {x['dataset'] for x in manifest} != set(EXPECTED):
        raise RuntimeError('Dataset inventory does not match all 16 expected datasets')
    save_json('datasets.json', manifest)
    binary = RUN/'LosslessBenchmark'
    shutil.copy2(str(SOURCE/'build/LosslessBenchmarkFull'), str(binary))
    config = dict(started=stamp(), runner_pid=os.getpid(), cpu=2, block_size=1000,
                  gef_uniform_partition_size=32000, codecs=CODECS, ranges=RANGES,
                  fauna_batch_size=65536,pcodec_chunk_size=262144,pcodec_page_size=262144,
                  binary_sha256=sha256(binary), binary=str(binary),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(SOURCE)).decode().strip(),
                  total_cases=len(manifest)*len(CODECS),
                  point_queries=dict(gef_neats_dac_leco=1000000,streaming_squash=100000,alp_buff_falcon=10000),
                  queries_per_range=dict(gef_neats_dac=10000,other=1000),
                  qualification='Existing baseline implementations and adapters; legacy validation_status remains not_checked. BUFF validates with its existing tolerance. See case logs.')
    save_json('config.json', config)
    with (RUN/'environment.txt').open('w') as envout:
        for command in [['uname','-a'],['lscpu'],['free','-h'],['df','-h',str(RUN),str(DATA)],
                        ['ldd',str(binary)],['git','status','--short'],
                        ['cat',str(SOURCE/'build/CMakeFiles/LosslessBenchmarkFull.dir/flags.make')],
                        ['cat',str(SOURCE/'build/CMakeFiles/gef_recent_compressors.dir/flags.make')]]:
            envout.write('\n$ ' + ' '.join(command) + '\n'); envout.flush()
            subprocess.call(command, cwd=str(SOURCE), stdout=envout, stderr=subprocess.STDOUT)
    status = []
    with (RUN/'results.txt').open('w', buffering=1) as results:
        writer = csv.DictWriter(results, fieldnames=FIELDS)
        writer.writeheader()
        # Small datasets first exercise every implementation before large inputs.
        for dataset in sorted(manifest, key=lambda x:(x['num_values'],x['dataset'])):
            for codec in CODECS:
                key = dataset['dataset'] + '__' + codec
                output = RUN/'cases'/(key+'.txt')
                metrics = RUN/'cases'/(key+'.metrics.txt')
                command = ['taskset','-c','2',str(binary),'-c',codec,'-b','1000',
                           '--fauna-batch-size','65536','--pcodec-chunk-size','262144','--pcodec-page-size','262144',
                           '-r',','.join(map(str,RANGES)),'-o',str(metrics),dataset['path']]
                item = dict(dataset=dataset['dataset'],codec=codec,status='RUNNING',
                            started=stamp(),command=command,log=str(output))
                status.append(item)
                log('START %d/%d %s' % (len(status),config['total_cases'],key))
                started = time.monotonic()
                with output.open('w', buffering=1) as caseout:
                    caseout.write('$ ' + ' '.join(command) + '\n'); caseout.flush()
                    proc = subprocess.Popen(command, stdout=caseout, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                    item['pid'] = proc.pid
                    save_json('status.json',status)
                    heartbeat = started
                    while proc.poll() is None:
                        time.sleep(1)
                        now = time.monotonic()
                        if now-heartbeat >= 60:
                            item['elapsed_seconds'] = round(now-started,2)
                            save_json('status.json',status)
                            log('RUNNING %s pid=%d elapsed=%.0fs' % (key,proc.pid,now-started))
                            heartbeat = now
                item.update(returncode=proc.returncode, finished=stamp(),
                            elapsed_seconds=round(time.monotonic()-started,2))
                content = output.read_text(errors='replace')
                print(content, end='' if content.endswith('\n') else '\n', flush=True)
                rows = []
                error = ''
                try:
                    if metrics.exists():
                        with metrics.open() as f:
                            rows = list(csv.DictReader(f))
                    if proc.returncode != 0:
                        raise ValueError('exit status %s' % proc.returncode)
                    if len(rows) != 1:
                        raise ValueError('expected exactly one metrics row')
                    row = rows[0]
                    if row['compressor'].lower() != codec or row['dataset'] != dataset['dataset']:
                        raise ValueError('wrong dataset or compressor in metrics')
                    if int(row['num_values']) != dataset['num_values'] or int(row['uncompressed_bits']) != dataset['num_values']*64:
                        raise ValueError('metrics do not cover the complete dataset')
                    required = FIELDS[:10]+['range_%d_mbs'%r for r in RANGES if r<dataset['num_values']]
                    for field in required[2:]:
                        if not math.isfinite(float(row[field])) or float(row[field]) <= 0:
                            raise ValueError('invalid numeric metric ' + field)
                    # Preserve measured metrics for validation warnings. The legacy
                    # harness reports first mismatches in logs and leaves the new
                    # validation fields not_checked; do not invent mismatch counts.
                    warning_lines = [line for line in content.splitlines() if
                                     'decompression error' in line.lower() or
                                     'warning:' in line.lower() or 'mismatch' in line.lower()]
                    if 'error:' in content.lower() and not warning_lines:
                        raise ValueError('benchmark reported a hard error')
                    item['status'] = 'WARNING' if warning_lines else 'OK'
                    item['validation_messages'] = warning_lines
                    item['validation_status'] = row.get('validation_status','not_checked')
                    writer.writerow(row)
                    results.flush()
                except Exception as exc:
                    error = str(exc)
                    unsupported = (codec in ['fauna','macaque'] and
                                   'upstream supports f32 only; input is not exactly representable' in content)
                    unsupported |= (codec == 'flea' and
                                    'block sum(abs(values)) exceeds safe imported implementation limit' in content)
                    item['status'] = 'UNSUPPORTED' if unsupported and proc.returncode == 1 and not rows else 'FAILED'
                    item['reason'] = error
                    item['details'] = content[-3000:]
                save_json('status.json',status)
                log('END %s status=%s seconds=%.1f %s' % (key,item['status'],item['elapsed_seconds'],error))
    summary = dict(finished=stamp(),total=len(status),
                   counts={state:sum(x['status']==state for x in status) for state in ['OK','WARNING','UNSUPPORTED','FAILED']})
    save_json('summary.json',summary)
    log('FINISHED ' + json.dumps(summary))
    return 1 if summary['counts']['FAILED'] else 0

if __name__ == '__main__':
    try:
        rc = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        rc = 2
    (RUN/'runner.exit').write_text(str(rc) + '\n')
    raise SystemExit(rc)
