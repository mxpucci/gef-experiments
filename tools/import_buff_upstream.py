from pathlib import Path
import re,hashlib,json
repo=Path(__file__).resolve().parents[1]
root=repo/'lib/buff/database/src'
records=[]
def item(file,needle):
 s=(root/file).read_text();p=s.index(needle);start=s.index('{',p);depth=0;i=start;state='code'
 while i<len(s):
  if state=='line':
   if s[i]=='\n':state='code'
  elif state=='string':
   if s[i]=='\\':i+=1
   elif s[i]=='"':state='code'
  elif state=='block':
   if s[i:i+2]=='*/':state='code';i+=1
  elif s[i:i+2]=='//':state='line';i+=1
  elif s[i:i+2]=='/*':state='block';i+=1
  elif s[i]=='"':state='string'
  elif s[i]=='{':depth+=1
  elif s[i]=='}':
   depth-=1
   if depth==0:
    x=s[p:i+1];records.append(dict(file=file,item=needle,sha256=hashlib.sha256(x.encode()).hexdigest()));return x
  i+=1
 raise ValueError(needle)
p='methods/prec_double.rs';b='methods/bit_packing.rs';s='compress/split_double.rs';m='compress/buff_simd.rs'
out='''#![allow(warnings)]
// Generated from the pinned authors repository by tools/import_buff_upstream.py.
// Codec bodies are original and unpatched (known upstream bugs are documented in
// docs/buff-leco-native.txt; none changes size or crashes on the benchmark datasets).
macro_rules! println { ($($t:tt)*) => {} }
macro_rules! print { ($($t:tt)*) => {} }
macro_rules! info { ($($t:tt)*) => {} }
macro_rules! trace { ($($t:tt)*) => {} }
macro_rules! warn { ($($t:tt)*) => {} }
use std::mem;
use std::slice::Iter;
use serde::{Serialize,Deserialize};
use itertools::Itertools;
use std::collections::HashMap;
use lazy_static::lazy_static;
pub struct Segment<'a,T>{ pub data:&'a [T] }
pub struct BorrowedBytes<'a>(pub &'a [u8]);
impl BorrowedBytes<'_> { pub fn as_slice(&self)->&[u8] { self.0 } }
impl<T> Segment<'_,T> {
'''+item('segment.rs','pub fn get_data(').replace('&Vec<T>','&[T]')+'\n}\n'
# Keep the authors' precision mapping verbatim.
c=(root/'compress/mod.rs').read_text();a=c.index('lazy_static!');z=c.index('\n\n\nlazy_static!',a);out+=c[a:z]+'\n'
for needle in ['pub const EXP_MASK:','pub const FIRST_ONE:','pub const NEG_ONE:']:
 t=(root/p).read_text();a=t.index(needle);out+=t[a:t.index(';',a)+1]+'\n'
out+=item(p,'pub struct PrecisionBound')+'\nimpl PrecisionBound {\n'
for needle in ['pub fn new(','pub fn set_length(','pub fn fetch_fixed_aligned(']:out+=item(p,needle)+'\n'
out+='}\n'+item(p,'pub fn get_precision_bound(')+'\n'
out+='pub const MAX_BITS:usize=32; pub const BYTE_BITS:usize=8;\n'
out+=item(b,'pub struct BitPack<B>')+'\n'
for needle in ['impl<B> BitPack<B>','impl<B: AsRef<[u8]>> BitPack<B>',"impl<'a> BitPack<&'a [u8]>","impl<'a> BitPack<&'a mut [u8]>",'impl Default for BitPack<Vec<u8>>','impl BitPack<Vec<u8>>']:
 out+=item(b,needle)+'\n'
out+=item('compress/buff_slice.rs','pub fn flip(')+'\n'
out+=item(s,'pub struct SplitBDDoubleCompress')+'\nimpl SplitBDDoubleCompress {\n'+item(s,'pub fn new(')+'\n'
for file,needle in [(s,'pub fn byte_fixed_encode'),(s,'pub fn byte_fixed_decode'),(m,'pub fn buff_decode_condition')]:
 x=item(file,needle)
 # Borrow compressed storage across the C ABI, without changing decoding logic.
 x=x.replace('bytes: Vec<u8>', "bytes: BorrowedBytes<'_>")
 out+=x+'\n'
out+='}\n'
(repo/'lib/buff-ffi/src/upstream.rs').write_text(out)
(repo/'lib/buff-ffi/upstream-items.json').write_text(json.dumps(records,indent=2)+'\n')
