mod upstream;
mod bit_vec_iter;
use upstream::*;
fn main(){
 for p in std::env::args().skip(1){
 let b=std::fs::read(&p).unwrap();let dec=u64::from_le_bytes(b[8..16].try_into().unwrap());
 let n=(b.len()-16)/16;
 let data:Vec<f64>=b[16+n*8..].chunks_exact(8).map(|v|f64::from_le_bytes(v.try_into().unwrap())).collect();
 let codec=SplitBDDoubleCompress::new(10,10,10usize.pow(dec as u32));
 let mut seg=Segment{data:&data};
 let plain=codec.byte_fixed_encode(&mut seg);
 let reference=codec.byte_fixed_decode(BorrowedBytes(&plain));
 let major=codec.buff_encode_majority(&mut seg);
 let major_size=major.len();let decoded=codec.buff_decode_majority(major);
 let mut times = Vec::new();
 for rep in 0..6 {
  // Alternate order, discard warm-up. Both encoders scan range themselves.
  let mut row=[0.0;4];
  for mode in if rep%2==0 {[0,1]} else {[1,0]} {
   let start=std::time::Instant::now();
   let bytes=if mode==0 {codec.byte_fixed_encode(&mut seg)} else {codec.buff_encode_majority(&mut seg)};
   std::hint::black_box(&bytes); row[mode]=start.elapsed().as_secs_f64();
   let start=std::time::Instant::now();
   let out=if mode==0 {codec.byte_fixed_decode(BorrowedBytes(&bytes))} else {codec.buff_decode_majority(bytes)};
   std::hint::black_box(&out); row[mode+2]=start.elapsed().as_secs_f64();
  }
  if rep>0 { times.push(row); }
 }
 let mut med=[0.0;4];
 for col in 0..4 { let mut a:Vec<f64>=times.iter().map(|r|r[col]).collect();a.sort_by(|a,b|a.partial_cmp(b).unwrap());med[col]=(n*8) as f64/1048576.0/a[2]; }
 let mismatches=reference.iter().zip(decoded.iter()).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
 println!("{},{},{},{},{},{},{},{},{},{}",p,n,plain.len(),major_size,mismatches,decoded.len(),med[0],med[1],med[2],med[3]);
 }
}
