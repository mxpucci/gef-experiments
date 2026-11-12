//! Thin C ABI over the authors' scalar BUFF methods. See upstream-items.json.
//! The generated module retains original codec bodies, with the documented
//! two-expression selective-decoder correction. No custom point decoder.
mod upstream;
use upstream::{BorrowedBytes,Segment,SplitBDDoubleCompress};
use std::{panic::{catch_unwind,AssertUnwindSafe},slice,ptr};
const HEADER:usize=16;
fn codec(scale:i64)->Option<SplitBDDoubleCompress>{
 let mut s=scale;let mut d=0;while s>1 && s%10==0{s/=10;d+=1;}
 if s!=1 || !(1..=15).contains(&d){return None;}
 Some(SplitBDDoubleCompress::new(10,10,scale as usize))
}
fn unpack(bytes:&[u8])->Option<(usize,SplitBDDoubleCompress,&[u8])>{
 if bytes.len()<HEADER+20{return None;}
 let n=u64::from_le_bytes(bytes[..8].try_into().ok()?) as usize;
 let scale=i64::from_le_bytes(bytes[8..16].try_into().ok()?);
 let body=&bytes[HEADER..];
 if u32::from_le_bytes(body[8..12].try_into().ok()?) as usize!=n{return None;}
 let ilen=u32::from_le_bytes(body[12..16].try_into().ok()?) as usize;
 let dlen=u32::from_le_bytes(body[16..20].try_into().ok()?) as usize;
 let width=ilen.checked_add(dlen)?;
 if width>64 || width==0{return None;}
 let needed=20usize.checked_add(n.checked_mul(width)?.checked_add(7)?/8)?;
 if body.len()<needed{return None;}
 Some((n,codec(scale)?,body))
}
#[no_mangle]
pub extern "C" fn buff_max_compressed_size(n:usize)->usize{n.saturating_mul(8).saturating_add(128)}
#[no_mangle]
pub unsafe extern "C" fn buff_compress_f64(input:*const f64,n:usize,output:*mut u8,capacity:usize,scale:i64)->i64{
 if input.is_null()||output.is_null()||n==0||n>u32::MAX as usize{return -1;}
 catch_unwind(AssertUnwindSafe(||->Option<i64>{
  let c=codec(scale)?;
  let mut seg=Segment{data:slice::from_raw_parts(input,n)};
  let payload=c.byte_fixed_encode(&mut seg);
  let total=HEADER.checked_add(payload.len())?;if total>capacity{return None;}
  ptr::copy_nonoverlapping((n as u64).to_le_bytes().as_ptr(),output,8);
  ptr::copy_nonoverlapping(scale.to_le_bytes().as_ptr(),output.add(8),8);
  ptr::copy_nonoverlapping(payload.as_ptr(),output.add(HEADER),payload.len());
  unpack(slice::from_raw_parts(output,total))?;
  Some(total as i64)
 })).ok().flatten().unwrap_or(-1)
}
#[no_mangle]
pub unsafe extern "C" fn buff_decompress_f64(input:*const u8,len:usize,output:*mut f64,n:usize)->i32{
 if input.is_null()||output.is_null(){return -1;}
 catch_unwind(AssertUnwindSafe(||->Option<()>{
  let (count,c,body)=unpack(slice::from_raw_parts(input,len))?;if n!=count{return None;}
  let values=c.byte_fixed_decode(BorrowedBytes(body));if values.len()!=n{return None;}
  ptr::copy_nonoverlapping(values.as_ptr(),output,n);Some(())
 })).ok().flatten().map_or(-1,|_|0)
}
#[no_mangle]
pub unsafe extern "C" fn buff_extract_f64(input:*const u8,len:usize,index:usize,output:*mut f64)->i32{
 buff_extract_range_f64(input,len,index,1,output)
}
#[no_mangle]
pub unsafe extern "C" fn buff_extract_range_f64(input:*const u8,len:usize,first:usize,count:usize,output:*mut f64)->i32{
 if input.is_null()||output.is_null(){return -1;}
 catch_unwind(AssertUnwindSafe(||->Option<()>{
  let (n,c,body)=unpack(slice::from_raw_parts(input,len))?;
  if first.checked_add(count)?>n{return None;}if count==0{return Some(());}
  let indices:Vec<usize>=(first..first+count).collect();
  let values=c.buff_decode_condition(BorrowedBytes(body),indices.iter());
  if values.len()!=count{return None;}
  ptr::copy_nonoverlapping(values.as_ptr(),output,count);Some(())
 })).ok().flatten().map_or(-1,|_|0)
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn original_selective_decoder_matches_full_at_non_byte_width() {
  let values:Vec<f64>=(0..4099).map(|i|((i*73%29003) as f64-9000.0)/10000.0).collect();
  let mut bytes=vec![0;buff_max_compressed_size(values.len())];
  let size=unsafe{buff_compress_f64(values.as_ptr(),values.len(),bytes.as_mut_ptr(),bytes.len(),10000)};
  assert!(size>0);bytes.truncate(size as usize);
  let mut full=vec![0.0;values.len()];
  assert_eq!(unsafe{buff_decompress_f64(bytes.as_ptr(),bytes.len(),full.as_mut_ptr(),full.len())},0);
  let width=u32::from_le_bytes(bytes[28..32].try_into().unwrap())+u32::from_le_bytes(bytes[32..36].try_into().unwrap());
  assert_ne!(width%8,0,"fixture must exercise partial-byte decoding");
  for (i,&v) in values.iter().enumerate(){
   let mut point=0.0;
   assert_eq!(unsafe{buff_extract_f64(bytes.as_ptr(),bytes.len(),i,&mut point)},0);
   assert_eq!(point.to_bits(),full[i].to_bits(),"index {}",i);
   assert!((full[i]-v).abs()<=0.00005);
  }
  for first in [0,7,999,4000]{
   let mut range=vec![0.0;37];
   assert_eq!(unsafe{buff_extract_range_f64(bytes.as_ptr(),bytes.len(),first,range.len(),range.as_mut_ptr())},0);
   assert!(range.iter().zip(full[first..].iter()).all(|(a,b)|a.to_bits()==b.to_bits()));
  }
 }
}
