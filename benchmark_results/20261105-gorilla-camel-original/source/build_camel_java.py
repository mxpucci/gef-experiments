#!/usr/bin/env python3
"""Build original Camel, using its documented maximum-decimal configuration.
Requires Java 8 JDK + Maven. No codec algorithm edits or precision extensions.
"""
from pathlib import Path
import argparse,os,shutil,subprocess
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,default=root/'build/camel-java');p.add_argument('--integer-bits',type=int,default=16,choices=range(3,32));a=p.parse_args()
build=a.build_dir.resolve();build.mkdir(parents=True,exist_ok=True)
source=root/'lib/camel-upstream';pin='c4b55c2226163b0bfb6c330aa78b7c0a97d872b3'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==pin
# Java 8 is the authors' required runtime/compiler.
env=os.environ.copy()
if os.path.exists('/usr/libexec/java_home'):env['JAVA_HOME']=subprocess.check_output(['/usr/libexec/java_home','-v','1.8'],text=True).strip()
java=Path(env['JAVA_HOME'])/'bin' if env.get('JAVA_HOME') else None
cp=build/'dependencies.txt'
subprocess.run(['mvn','-q','-f',str(source/'pom.xml'),'dependency:build-classpath','-Dmdep.outputFile='+str(cp)],env=env,check=True)
src=build/'src';shutil.copytree(source/'src/main/java',src,dirs_exist_ok=True)
f=src/'yyy/ts/compress/camel/Camel.java';s=f.read_text();assert s.count('DECIMAL_MAX_COUNT = 3')==1
s=s.replace('DECIMAL_MAX_COUNT = 3','DECIMAL_MAX_COUNT = 4')
assert s.count('out.writeInt(diff, 16)')==1 and s.count('size += 17;')==1
s=s.replace('out.writeInt(diff, 16)',f'out.writeInt(diff, {a.integer_bits})').replace('size += 17;',f'size += {a.integer_bits+1};')
f.write_text(s)
f=src/'yyy/ts/compress/camel/CamelDecompressor.java';s=f.read_text();assert s.count('diffVal = in.readInt(16);')==1
f.write_text(s.replace('diffVal = in.readInt(16);',f'diffVal = in.readInt({a.integer_bits});'))
classes=build/'classes';classes.mkdir(exist_ok=True)
subprocess.run([str(java/'javac') if java else 'javac','-encoding','UTF-8','-proc:none','-cp',cp.read_text().strip(),'-sourcepath',str(src),'-d',str(classes)]+[str(src/'yyy/ts/compress/camel'/f) for f in ['Camel.java','CamelDecompressor.java','SourceBTree.java']]+[str(root/'benchmark/CamelOriginal/CamelBenchmark.java')],env=env,check=True)
# These are the only external classes reached by the compression harness.
jars=[Path(x) for x in cp.read_text().strip().split(os.pathsep) if Path(x).name.startswith(('dsiutils-','fastutil-','slf4j-api-'))]
assert len(jars)==3
lib=build/'jars';lib.mkdir(exist_ok=True)
for jar in jars:shutil.copy2(jar,lib/jar.name)
(build/'classpath.txt').write_text(os.pathsep.join(map(str,[classes]+[lib/j.name for j in jars]))+'\n')
(build/'PROVENANCE.txt').write_text('https://github.com/yoyo185644/camel\n'+pin+f'\nSource configuration: DECIMAL_MAX_COUNT 3 -> 4; integer difference width 16 -> {a.integer_bits} (README configuration); matching size counter.\n')
print(build/'classpath.txt')
