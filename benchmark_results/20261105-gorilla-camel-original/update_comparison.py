"""Update the existing comparison only after all 16 fresh measurements exist."""
from pathlib import Path
import csv,json,hashlib,shutil,datetime,subprocess
root=Path(__file__).resolve().parent
plot=Path('/Users/michelangelopucci/gef-performance/experiments/results/engineering/competitors-prepool-20261105')
assert json.loads((root/'complete.json').read_text())['rows']==32
fresh=list(csv.DictReader((root/'results.csv').open()))
assert len(fresh)==32 and len({(r['compressor'],r['dataset']) for r in fresh})==32
assert all(r['validation_status'] in ('passed','bitwise_passed','bitwise_failed') for r in fresh)
lookup={(r['compressor'].lower(),r['dataset']):r for r in fresh}
backup=plot/('before-original-gorilla-camel-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));backup.mkdir()
for p in plot.iterdir():
 if p.is_file():shutil.copy2(p,backup/p.name)
rows=list(csv.DictReader((plot/'competitors-snapshot.csv').open()))
changes=[]
for row in rows:
 key=(row['requested_compressor'],row['dataset'])
 if key not in lookup:continue
 new=lookup[key];assert int(row['num_values'])==int(new['num_values'])
 changes.append(dict(dataset=row['dataset'],compressor=row['compressor'],old_bits=int(row['compressed_bits']),new_bits=int(new['compressed_bits']),validation=new['validation_status']))
 for name,value in new.items():row[name]=value
 row['owned_bytes']='';row['measurement_run']='gef-experiments/20261105-gorilla-camel-original'
 row['audit_status']='full_bitwise_passed' if new['roundtrip_mismatches']=='0' else 'full_bitwise_failed'
with (plot/'competitors-snapshot.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
shutil.copy2(root/'results.csv',plot/'gorilla-camel-original-full16.csv')
p=plot/'compare.py';s=p.read_text()
s=s.replace("data = pd.concat([c[columns], g[columns]], ignore_index=True)","# Falcon excluded by user; retain historical raw rows only.\nc = c[c.requested_compressor != 'falcon'].copy()\ndata = pd.concat([c[columns], g[columns]], ignore_index=True)")
s=s.replace("row = dict(label=label, datasets=len(group), qualification=group.qualification.iloc[0])", "qualification = ('bitwise audit failed' if 'bitwise audit failed' in set(group.qualification) else group.qualification.iloc[0])\n        row = dict(label=label, datasets=len(group), qualification=qualification)")
s=s.replace('ALP rerun 2026-11-05; other measurements retained.', 'ALP/Gorilla/Camel rerun 2026-11-05; original Java Camel, warmed JVM.')
s=s.replace('‡ Gorilla/Camel/Falcon: failed bitwise audit. Hollow points excluded.', '‡ Original Camel: not lossless on some inputs. Hollow points excluded; Falcon omitted.')
s=s.replace("ROOT/'alp-corrected-full16.csv', Path(__file__)","ROOT/'alp-corrected-full16.csv', ROOT/'gorilla-camel-original-full16.csv', Path(__file__)")
s=s.replace("c.audit_status == 'sample_bitwise_failed'", "c.audit_status.isin(['sample_bitwise_failed', 'full_bitwise_failed'])")
s=s.replace('LosslessBenchmarkFull; single measurement; ', 'LosslessBenchmark family; single measurement; ')
p.write_text(s)
provenance=dict(backup=str(backup),source=str(root),source_sha256=hashlib.sha256((root/'results.csv').read_bytes()).hexdigest(),changes=changes,excluded=['Falcon'],note='Camel remains outside aggregate lossless frontiers if any dataset fails; no imputation or averaging over a smaller subset.')
(plot/'gorilla-camel-update-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
subprocess.run(['/Users/michelangelopucci/miniconda3/envs/venv/bin/python',str(p)],check=True)
