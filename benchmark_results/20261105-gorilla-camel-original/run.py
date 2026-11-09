from pathlib import Path
import subprocess,csv,json,datetime,hashlib,os
root=Path(__file__).resolve().parent
repo=root.parents[1]
files=sorted(Path('/dev/shm/testPF-gef-20261103/extracted/datasets').glob('*.bin'),key=lambda p:p.stat().st_size)
assert len(files)==16
(root/'cases').mkdir(exist_ok=True)
status=[];rows=[]
for p in files:
 info={'dataset':p.stem,'status':'running','started':datetime.datetime.now().isoformat()};status.append(info)
 (root/'status.json').write_text(json.dumps(status,indent=2))
 out=root/'cases'/(p.stem+'.csv')
 env=os.environ.copy()
 if p.stem=='LAT':
  env['GEF_CAMEL_CLASSPATH']=(repo/'build/camel-java-int19/classpath.txt').read_text().strip()
 info['camel_integer_bits']=19 if p.stem=='LAT' else 16
 cmd=['taskset','-c','8',str(root/'LosslessBenchmark'),str(p),'-c','gorilla,camel','-r','10,100,1000,10000,100000','-o',str(out)]
 if out.exists() and len(list(csv.DictReader(out.open())))==2:
  r=subprocess.CompletedProcess(cmd,0);info['reused_existing_result']=True
 else:
  with (root/'cases'/(p.stem+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
 parsed=list(csv.DictReader(out.open())) if out.exists() else []
 info.update(returncode=r.returncode,finished=datetime.datetime.now().isoformat(),status='completed' if r.returncode==0 and len(parsed)==2 else 'error',validation={x['compressor']:x['validation_status'] for x in parsed})
 rows+=parsed
 if rows:
  with (root/'results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(key for row in rows for key in row)));w.writeheader();w.writerows(rows)
 (root/'status.json').write_text(json.dumps(status,indent=2))
 print(p.stem,info['status'],info['validation'],flush=True)
(root/'complete.json').write_text(json.dumps({'datasets':len(status),'rows':len(rows),'binary_sha256':hashlib.sha256((root/'LosslessBenchmark').read_bytes()).hexdigest()},indent=2))
