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
// Codec bodies are original except the GEF-PATCH-marked fixes applied by the import tool:
// three encoder bugs (min/max, power-of-two width, narrow-range underflow) and the
// selective decoder's extra shift of the trailing low bits.
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
 if needle=='pub fn byte_fixed_encode':
  # Minimal encoder bug fixes (same logic as the May 2026 adapter fixes on main,
  # commits d50077d3 and 83a14323). Each replacement is asserted to apply once.
  # Every block upstream encodes correctly is encoded byte-identically.
  for old,rep in [
   # GEF-PATCH 1: upstream's `else if` means the first value can never set max
   # (e.g. [10,5,7,6] at scale 10 decodes as [6,5,5,6]).
   ('''            else if fixed>max {''',
    '''            // GEF-PATCH 1: was `else if`; the first value must also update max.
            if fixed>max {'''),
   # GEF-PATCH 2: log2().ceil() is one bit short when delta is a power of two
   # (e.g. [1,2,3,17] decodes as [1,2,3,1]); use the exact bit length.
   ('''            cal_int_length = (delta as f64).log2().ceil();''',
    '''            // GEF-PATCH 2: was `(delta as f64).log2().ceil()`, one bit short for powers of two.
            cal_int_length = (64 - (delta as u64).leading_zeros()) as f64;'''),
   # GEF-PATCH 3: `fixed_len - dec_len` underflows for narrow ranges
   # (delta < 2^dec_len); pad the width up to dec_len.
   ('''        let fixed_len = cal_int_length as usize;
        bound.set_length((cal_int_length as u64-dec_len), dec_len);''',
    '''        // GEF-PATCH 3: was `cal_int_length as usize`; pad to dec_len so `fixed_len - dec_len` cannot underflow.
        let fixed_len = std::cmp::max(cal_int_length as usize, dec_len as usize);
        bound.set_length((fixed_len as u64-dec_len), dec_len);'''),
  ]:
   assert x.count(old)==1,old
   x=x.replace(old,rep)
 if needle=='pub fn buff_decode_condition':
  old='((cur as u64)<< remain as u64)'
  assert x.count(old)==2
  # GEF-PATCH 4: the trailing `remain` bits are the low bits; upstream shifted them
  # left by `remain` again (selective decode then disagrees with full decode).
  x=x.replace(old,'(cur as u64) /* GEF-PATCH 4: was `(cur as u64)<< remain as u64`; trailing bits are the low bits */')
 out+=x+'\n'
out+='}\n'
(repo/'lib/buff-ffi/src/upstream.rs').write_text(out)
(repo/'lib/buff-ffi/upstream-items.json').write_text(json.dumps(records,indent=2)+'\n')
