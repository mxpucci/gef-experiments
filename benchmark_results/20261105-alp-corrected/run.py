from pathlib import Path
import subprocess,csv,json,datetime,hashlib
root=Path(__file__).resolve().parent
files=sorted(Path('/dev/shm/testPF-gef-20261103/extracted/datasets').glob('*.bin'),key=lambda p:p.stat().st_size)
assert len(files)==16
(root/'cases').mkdir(exist_ok=True)
status=[];rows=[]
for p in files:
 info={'dataset':p.stem,'status':'running','started':datetime.datetime.now().isoformat()}
 status.append(info);(root/'status.json').write_text(json.dumps(status,indent=2))
 out=root/'cases'/(p.stem+'.csv')
 cmd=['taskset','-c','8',str(root/'LosslessBenchmark'),str(p),'-c','alp','-b','1000','-r','10,100,1000,10000,100000','-o',str(out)]
 with (root/'cases'/(p.stem+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 info.update(returncode=r.returncode,finished=datetime.datetime.now().isoformat())
 parsed=list(csv.DictReader(out.open())) if out.exists() else []
 info['status']='passed' if r.returncode==0 and len(parsed)==1 and parsed[0]['validation_status']=='passed' else 'failed'
 rows+=parsed
 if rows:
  with (root/'results.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (root/'status.json').write_text(json.dumps(status,indent=2))
 print(p.stem,info['status'],flush=True)
(root/'complete.json').write_text(json.dumps({'cases':len(status),'passed':sum(x['status']=='passed' for x in status),'binary_sha256':hashlib.sha256((root/'LosslessBenchmark').read_bytes()).hexdigest()},indent=2))
