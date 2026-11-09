//! Thin C ABI over the authors' scalar BUFF methods. See upstream-items.json.
//! This is plain BUFF (`byte_fixed_encode`, one column per dataset), not the
//! BUFF-major / sparse-outlier variants from the paper.
//! The generated module retains original codec bodies, with the documented
//! GEF-PATCH-marked fixes listed in tools/import_buff_upstream.py (three
//! encoder bugs and the selective decoder's trailing-bit shift). No custom
//! point decoder.
mod upstream;
use upstream::{BorrowedBytes,Segment,SplitBDDoubleCompress,PRECISION_MAP};
use std::{panic::{catch_unwind,AssertUnwindSafe},slice,ptr};
const HEADER:usize=16;
// Upstream PRECISION_MAP maps 13..=15 decimals to 10 fractional bits (an error
// of ~1e-3), so only 1..=12 decimals keep BUFF's precision bound.
const MAX_DECIMALS:i32=12;
fn decimals(scale:i64)->Option<i32>{
 let mut s=scale;let mut d=0;while s>1 && s%10==0{s/=10;d+=1;}
 if s!=1 || !(1..=MAX_DECIMALS).contains(&d){return None;}
 // Upstream derives the precision as `(scale as f32).log10() as i32`.
 if (scale as f32).log10() as i32!=d{return None;}
 Some(d)
}
fn codec(scale:i64)->Option<SplitBDDoubleCompress>{
 decimals(scale)?;Some(SplitBDDoubleCompress::new(10,10,scale as usize))
}
// Upstream's fixed-point conversion shifts by `63-exp-dec_len`; reject inputs
// whose fixed-point value would not fit (and NaN/inf) instead of wrapping.
fn representable(values:&[f64],scale:i64)->Option<()>{
 let dlen=*PRECISION_MAP.get(&decimals(scale)?)?;
 let limit=2f64.powi(62-dlen);
 if values.iter().all(|v|v.is_finite() && v.abs()<limit){Some(())}else{None}
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
  representable(slice::from_raw_parts(input,n),scale)?;
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
 fn roundtrip(values:&[f64],scale:i64)->(Vec<u8>,Vec<f64>){
  let mut bytes=vec![0;buff_max_compressed_size(values.len())];
  let size=unsafe{buff_compress_f64(values.as_ptr(),values.len(),bytes.as_mut_ptr(),bytes.len(),scale)};
  assert!(size>0,"compress failed");bytes.truncate(size as usize);
  let mut full=vec![0.0;values.len()];
  assert_eq!(unsafe{buff_decompress_f64(bytes.as_ptr(),bytes.len(),full.as_mut_ptr(),full.len())},0);
  let tol=0.5/scale as f64;
  for (i,(&v,&d)) in values.iter().zip(&full).enumerate(){assert!((v-d).abs()<tol,"index {}: {} decoded as {}",i,v,d);}
  for i in 0..values.len(){
   let mut point=0.0;
   assert_eq!(unsafe{buff_extract_f64(bytes.as_ptr(),bytes.len(),i,&mut point)},0);
   assert_eq!(point.to_bits(),full[i].to_bits(),"point {}",i);
  }
  for first in 0..values.len(){for count in 1..=(values.len()-first).min(5){
   let mut range=vec![0.0;count];
   assert_eq!(unsafe{buff_extract_range_f64(bytes.as_ptr(),bytes.len(),first,count,range.as_mut_ptr())},0);
   assert!(range.iter().zip(&full[first..]).all(|(a,b)|a.to_bits()==b.to_bits()),"range {}+{}",first,count);
  }}
  (bytes,full)
 }
 fn width(bytes:&[u8])->u32{u32::from_le_bytes(bytes[28..32].try_into().unwrap())+u32::from_le_bytes(bytes[32..36].try_into().unwrap())}
 #[test]
 fn first_value_can_be_the_maximum(){ // GEF-PATCH 1; upstream decoded [6,5,5,6]
  roundtrip(&[1.0,0.5,0.7,0.6],10);
  roundtrip(&[10.0,5.0,7.0,6.0],10);
 }
 #[test]
 fn power_of_two_delta(){ // GEF-PATCH 2; upstream decoded [1,2,3,1]
  roundtrip(&[1.0,2.0,3.0,17.0],10);
  for k in 0..20{roundtrip(&[0.0,(1u64<<k) as f64/32.0],10);}
 }
 #[test]
 fn narrow_and_constant_ranges(){ // GEF-PATCH 3; upstream underflowed fixed_len-dec_len
  roundtrip(&[3.14],100);
  roundtrip(&[3.14,3.14,3.14],100);
  roundtrip(&[3.14,3.15,3.14],100);
  roundtrip(&[-0.001,0.0,0.001],1000);
 }
 #[test]
 fn every_supported_precision_and_width(){
  for d in 1..=MAX_DECIMALS{
   let scale=10i64.pow(d as u32);
   for span in [1i64,3,10,255,256,1000,65537,1<<20]{
    let values:Vec<f64>=(0..37).map(|i|((i*7919)%(span+1)-span/2) as f64/scale as f64).collect();
    let (bytes,_)=roundtrip(&values,scale);
    assert!(width(&bytes)<=64);
   }
  }
 }
 #[test]
 fn rejects_inexact_precisions_and_unrepresentable_values(){
  let mut out=vec![0u8;256];
  for scale in [1i64,7,20,10i64.pow(13),10i64.pow(15)]{
   assert_eq!(unsafe{buff_compress_f64([1.0f64].as_ptr(),1,out.as_mut_ptr(),out.len(),scale)},-1,"scale {}",scale);
  }
  for v in [f64::NAN,f64::INFINITY,1e300,1e5]{ // 1e5 needs 17+50 bits at 12 decimals
   assert_eq!(unsafe{buff_compress_f64([v].as_ptr(),1,out.as_mut_ptr(),out.len(),10i64.pow(12))},-1,"value {}",v);
  }
 }
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
