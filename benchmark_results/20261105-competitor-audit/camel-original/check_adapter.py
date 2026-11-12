from pathlib import Path
import csv,subprocess,struct,json,tempfile,sys
root=Path(__file__).resolve().parents[3]
cp=(root/'build/camel-java/classpath.txt').read_text().strip()
reference={Path(r[0]).stem:int(r[3]) for r in csv.reader((Path(__file__).parent/'configured4.csv').open())}
rows=[]
with tempfile.TemporaryDirectory(prefix='gef-camel-check-') as tmp:
 tmp=Path(tmp)
 for path in sorted(Path(sys.argv[1]).glob('*.bin')):
  b=path.read_bytes();n=(len(b)-16)//16
  points=[0,n-1]+([999,1000] if n>1000 else [])
  payload=struct.pack('<Q',n)+b[16+n*8:]+struct.pack('<Q',len(points))+b''.join(struct.pack('<Q',p) for p in points)
  # Include a cross-block range, a tail range, and a single-value range.
  ranges=[(10,[995,n-10]),(1,[0,n-1])] if n>=1010 else [(1,[0,n-1])]
  payload+=struct.pack('<Q',len(ranges))
  for size,starts in ranges:payload+=struct.pack('<QQ',size,len(starts))+b''.join(struct.pack('<Q',p) for p in starts)
  (tmp/'in').write_bytes(payload)
  subprocess.run(['java','-Xmx2g','-cp',cp,'CamelBenchmark',str(tmp/'in'),str(tmp/'out')],check=True)
  tokens=(tmp/'out').read_text().splitlines()[0].split();actual=int(tokens[3]);assert actual==reference[path.stem],(path,actual,reference[path.stem])
  rows.append(dict(sample=path.stem,n=n,compressed_bits=int(tokens[0]),mismatches=actual))
(Path(__file__).parent/'adapter-check.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Original vs adapter mismatch counts agree on',len(rows),'samples')
