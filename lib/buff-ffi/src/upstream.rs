#![allow(warnings)]
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
pub fn get_data(&self) ->  &[T]
	{
		&self.data
	}
}
lazy_static! {
    pub static ref PRECISION_MAP: HashMap<i32, i32> =[(1, 5),
        (2, 8),
        (3, 11),
        (4, 15),
        (5, 18),
        (6, 21),
        (7, 25),
        (8, 28),
        (9, 31),
        (10, 35),
        (11, 38),
        (12, 50),
        (13, 10),
        (14, 10),
        (15, 10)]
        .iter().cloned().collect();
}
pub const EXP_MASK: u64 = 0b0111111111110000000000000000000000000000000000000000000000000000;
pub const FIRST_ONE: u64 = 0b1000000000000000000000000000000000000000000000000000000000000000;
pub const NEG_ONE: u64 = 0b1111111111111111111111111111111111111111111111111111111111111111;
pub struct PrecisionBound {
    position: u64,
    precision: f64,
    precision_exp : i32,
    int_length: u64,
    decimal_length: u64,
}
impl PrecisionBound {
pub fn new(precision:f64) -> Self {
        let mut e = PrecisionBound { position: 0, precision: precision, precision_exp : 0, int_length: 0, decimal_length: 0 };
        let xu = unsafe { mem::transmute::<f64, u64>(precision) };
        e.precision_exp = ((xu & EXP_MASK) >> 52) as i32 - 1023 as i32;
        e
    }
pub fn set_length(&mut self, ilen:u64, dlen:u64){
        self.decimal_length = dlen;
        self.int_length = ilen;
    }
pub fn fetch_fixed_aligned(&self, bd:f64) -> i64{
        let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
        let exp = ((bdu & EXP_MASK) >> 52) as i32 - 1023 as i32;
        let sign = bdu&FIRST_ONE;
        let mut fixed = 0u64;
        if exp<self.precision_exp{
            fixed = 0u64;
        }else{
            fixed = ((bdu << (11)) | FIRST_ONE) >> (63-exp-self.decimal_length as i32) as u64;
            if sign!=0{
                fixed = !(fixed-1);
            }
        }
        let signed_int = unsafe { mem::transmute::<u64, i64>(fixed) };
        signed_int
    }
}
pub fn get_precision_bound(precision: i32) -> f64{
    let mut str = String::from("0.");
    for pos in 0..precision{
        str.push('0');
    }
    str.push_str("49");
    let error = str.parse().unwrap();
    error
}
pub const MAX_BITS:usize=32; pub const BYTE_BITS:usize=8;
pub struct BitPack<B> {
    buff: B,
    cursor: usize,
    bits: usize
}
impl<B> BitPack<B> {
    #[inline]
    pub fn new(buff: B) -> Self {
        BitPack { buff: buff, cursor: 0, bits: 0 }
    }

    #[inline]
    pub fn sum_bits(&self) -> usize {
        self.cursor * BYTE_BITS + self.bits
    }

    #[inline]
    pub fn with_cursor(&mut self, cursor: usize) -> &mut Self {
        self.cursor = cursor;
        self
    }

    #[inline]
    pub fn with_bits(&mut self, bits: usize) -> &mut Self {
        self.bits = bits;
        self
    }
}
impl<B: AsRef<[u8]>> BitPack<B> {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.buff.as_ref()
    }
}
impl<'a> BitPack<&'a [u8]> {
    /// ```
    /// use bitpack::BitPack;
    /// use time_series_start::methods::bit_packing::BitPack;
    ///
    /// let mut buff = [218, 255];
    ///
    /// let mut bitpack = BitPack::<&[u8]>::new(&buff);
    /// assert_eq!(bitpack.read(4).unwrap(), 10);
    /// assert_eq!(bitpack.read(10).unwrap(), 1021);
    /// assert_eq!(bitpack.read(2).unwrap(), 3);
    /// ```
    pub fn read(&mut self, mut bits: usize) -> Result<u32, usize> {
        if bits > MAX_BITS || self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(bits);
        };

        let mut bits_left = 0;
        let mut output = 0;
        loop {
            let byte_left = BYTE_BITS - self.bits;

            if bits <= byte_left {
                let mut bb = self.buff[self.cursor] as u32;
                bb >>= self.bits as u32;
                bb &= ((1 << bits) - 1) as u32;
                output |= bb << bits_left;
                self.bits += bits;
                break
            }

            let mut bb = self.buff[self.cursor] as u32;
            bb >>= self.bits as u32;
            bb &= ((1 << byte_left) - 1) as u32;
            output |= bb << bits_left;
            self.bits += byte_left;
            bits_left += byte_left as u32;
            bits -= byte_left;

            if self.bits >= BYTE_BITS {
                self.cursor += 1;
                self.bits -= BYTE_BITS;
            }
        }
        Ok(output)
    }

    /***
    read bits less than BYTE_BITS
     */

    #[inline]
    pub fn read_bits(&mut self, mut bits: usize) -> Result<u8, usize> {
        if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            println!("buff length: {}, cursor: {}, bits: {}", self.buff.len(), self.cursor,self.bits);
            return Err(bits);
        };

        let mut bits_left = 0;
        let mut output = 0;
        loop {
            let byte_left = BYTE_BITS - self.bits;

            if bits <= byte_left {
                let mut bb = self.buff[self.cursor] as u32;
                bb >>= self.bits as u32;
                bb &= ((1 << bits) - 1) as u32;
                output |= bb << bits_left;
                self.bits += bits;
                break
            }

            let mut bb = self.buff[self.cursor] as u32;
            bb >>= self.bits as u32;
            bb &= ((1 << byte_left) - 1) as u32;
            output |= bb << bits_left;
            self.bits += byte_left;
            bits_left += byte_left as u32;
            bits -= byte_left;

            if self.bits >= BYTE_BITS {
                self.cursor += 1;
                self.bits -= BYTE_BITS;
            }
        }
        Ok(output as u8)
    }

    #[inline]
    pub fn finish_read_byte(&mut self){
        self.cursor += 1;
        self.bits = 0;
        // println!("cursor now at {}" , self.cursor)
    }

    #[inline]
    pub fn read_byte(&mut self) -> Result<u8, usize> {
        self.cursor += 1;
        let output = self.buff[self.cursor] as u8;
        Ok(output)
    }

    #[inline]
    pub fn read_n_byte(&mut self,n:usize) -> Result<&[u8], usize> {
        self.cursor += 1;
        let end = self.cursor+n;
        let output = &self.buff[self.cursor..end];
        self.cursor += n-1;
        Ok(output)
    }

    #[inline]
    pub fn read_n_byte_unmut(&self,start:usize, n:usize) -> Result<&[u8], usize> {
        let s = start+self.cursor + 1;
        let end =s+n;
        let output = &self.buff[s..end];
        Ok(output)
    }

    #[inline]
    pub fn skip_n_byte(&mut self, mut n: usize) -> Result<(), usize> {
        self.cursor += n;
        // println!("current cursor{}, current bits:{}",self.cursor,self.bits);
        Ok(())
    }
    #[inline]
    pub fn skip(&mut self, mut bits: usize) -> Result<(), usize> {
        if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(bits);
        };
        // println!("current cursor{}, current bits:{}",self.cursor,self.bits);
        // println!("try to skip {} bits",bits);
        let bytes = bits/BYTE_BITS;
        let left = bits%BYTE_BITS;

        let cur_bits = (self.bits +left);
        self.cursor = self.cursor + bytes + cur_bits/BYTE_BITS;
        self.bits = cur_bits%BYTE_BITS;

        // println!("current cursor{}, current bits:{}",self.cursor,self.bits);
        Ok(())
    }
}
impl<'a> BitPack<&'a mut [u8]> {
    /// ```
    /// use bitpack::BitPack;
    /// use time_series_start::methods::bit_packing::BitPack;
    ///
    /// let mut buff = [0; 2];
    ///
    /// {
    ///     let mut bitpack = BitPack::<&mut [u8]>::new(&mut buff);
    ///     bitpack.write(10, 4).unwrap();
    ///     bitpack.write(1021, 10).unwrap();
    ///     bitpack.write(3, 2).unwrap();
    /// }
    ///
    /// assert_eq!(buff, [218, 255]);
    /// ```

    pub fn write(&mut self, mut value: u32, mut bits: usize) -> Result<(), usize> {
        if bits > MAX_BITS || self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(bits);
        }
        if bits < MAX_BITS {
            value &= ((1 << bits) - 1) as u32;
        }

        loop {
            let bits_left = BYTE_BITS - self.bits;

            if bits <= bits_left {
                self.buff[self.cursor] |= (value as u8) << self.bits as u8;
                self.bits += bits;

                if self.bits >= BYTE_BITS {
                    self.cursor += 1;
                    self.bits = 0;
                }

                break
            }

            let bb = value & ((1 << bits_left) - 1) as u32;
            self.buff[self.cursor] |= (bb as u8) << self.bits as u8;
            self.cursor += 1;
            self.bits = 0;
            value >>= bits_left as u32;
            bits -= bits_left;
        }
        Ok(())
    }

    /***
    read bits less then BYTE_BITS
     */
    pub fn write_bits(&mut self, mut value: u32, mut bits: usize) -> Result<(), usize> {
        if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(bits);
        }
        value &= ((1 << bits) - 1) as u32;

        loop {
            let bits_left = BYTE_BITS - self.bits;

            if bits <= bits_left {
                self.buff[self.cursor] |= (value as u8) << self.bits as u8;
                self.bits += bits;

                if self.bits >= BYTE_BITS {
                    self.cursor += 1;
                    self.bits = 0;
                }

                break
            }

            let bb = value & ((1 << bits_left) - 1) as u32;
            self.buff[self.cursor] |= (bb as u8) << self.bits as u8;
            self.cursor += 1;
            self.bits = 0;
            value >>= bits_left as u32;
            bits -= bits_left;
        }
        Ok(())
    }
}
impl Default for BitPack<Vec<u8>> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}
impl BitPack<Vec<u8>> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(Vec::with_capacity(capacity))
    }

    /// ```
    /// use bitpack::BitPack;
    /// use time_series_start::methods::bit_packing::BitPack;
    ///
    /// let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(2);
    /// bitpack_vec.write(10, 4).unwrap();
    /// bitpack_vec.write(1021, 10).unwrap();
    /// bitpack_vec.write(3, 2).unwrap();
    ///
    /// assert_eq!(bitpack_vec.as_slice(), [218, 255]);
    /// #
    /// # let mut bitpack = BitPack::<&[u8]>::new(bitpack_vec.as_slice());
    /// # assert_eq!(bitpack.read(4).unwrap(), 10);
    /// # assert_eq!(bitpack.read(10).unwrap(), 1021);
    /// # assert_eq!(bitpack.read(2).unwrap(), 3);
    /// ```
    #[inline]
    pub fn write(&mut self, value: u32, bits: usize) -> Result<(), usize> {
        let len = self.buff.len();

        if let Some(bits) = (self.sum_bits() + bits).checked_sub(len * BYTE_BITS) {
            self.buff.resize(len + (bits + BYTE_BITS - 1) / BYTE_BITS, 0x0);
        }

        let mut bitpack = BitPack {
            buff: self.buff.as_mut_slice(),
            cursor: self.cursor,
            bits: self.bits
        };

        bitpack.write(value, bits)?;

        self.bits = bitpack.bits;
        self.cursor = bitpack.cursor;

        Ok(())
    }

    /***
    read bits less then BYTE_BITS
     */
    #[inline]
    pub fn write_bits(&mut self, value: u32, bits: usize) -> Result<(), usize> {
        let len = self.buff.len();

        if let Some(bits) = (self.sum_bits() + bits).checked_sub(len * BYTE_BITS) {
            self.buff.resize(len + (bits + BYTE_BITS - 1) / BYTE_BITS, 0x0);
        }

        let mut bitpack = BitPack {
            buff: self.buff.as_mut_slice(),
            cursor: self.cursor,
            bits: self.bits
        };

        bitpack.write_bits(value, bits)?;

        self.bits = bitpack.bits;
        self.cursor = bitpack.cursor;

        Ok(())
    }



    #[inline]
    pub fn write_bytes(&mut self, value: &mut Vec<u8>) -> Result<(), usize> {
        self.buff.append(value);
        Ok(())
    }

    #[inline]
    pub fn write_byte(&mut self, value: u8) -> Result<(), usize> {
        self.buff.push(value);
        Ok(())
    }

    #[inline]
    pub fn finish_write_byte(&mut self) {
        let len = self.buff.len();
        self.buff.resize(len + 1, 0x0);
        self.bits = 0;
        self.cursor = len;
        // println!("cursor now at {}" , self.cursor)
    }

    #[inline]
    pub fn into_vec(self) -> Vec<u8> {
        // println!("buff length: {}, cursor: {}, bits: {}", self.buff.len(), self.cursor,self.bits);
        self.buff
    }
}
pub fn flip(x:u8)->u8{
    let offset = 1u8<< 7;
    x^offset
}
pub struct SplitBDDoubleCompress {
    chunksize: usize,
    batchsize: usize,
    pub(crate) scale: usize
}
impl SplitBDDoubleCompress {
pub fn new(chunksize: usize, batchsize: usize, scale: usize) -> Self {
        SplitBDDoubleCompress { chunksize, batchsize, scale}
    }
pub fn byte_fixed_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let mut fixed_vec = Vec::new();

        let mut t:u32 = seg.get_data().len() as u32;
        let mut prec = 0;
        if self.scale == 0{
            prec = 0;
        }
        else{
            prec = (self.scale as f32).log10() as i32;
        }
        let prec_delta = get_precision_bound(prec);
        println!("precision {}, precision delta:{}", prec, prec_delta);

        let mut bound = PrecisionBound::new(prec_delta);
        // let start1 = Instant::now();
        let dec_len = *(PRECISION_MAP.get(&prec).unwrap()) as u64;
        bound.set_length(0,dec_len);
        let mut min = i64::max_value();
        let mut max = i64::min_value();

        for bd in seg.get_data(){
            let fixed = bound.fetch_fixed_aligned((*bd).into());
            if fixed<min {
                min = fixed;
            }
            else if fixed>max {
                max = fixed;
            }
            fixed_vec.push(fixed);
        }
        let delta = max-min;
        let base_fixed = min;
        println!("base integer: {}, max:{}",base_fixed,max);
        let ubase_fixed = unsafe { mem::transmute::<i64, u64>(base_fixed) };
        let base_fixed64:i64 = base_fixed;
        let mut single_val = false;
        let mut cal_int_length = 0.0;
        if delta == 0 {
            single_val = true;
        }else {
            cal_int_length = (delta as f64).log2().ceil();
        }

        let fixed_len = cal_int_length as usize;
        bound.set_length((cal_int_length as u64-dec_len), dec_len);
        let ilen = fixed_len -dec_len as usize;
        let dlen = dec_len as usize;
        println!("int_len:{},dec_len:{}",ilen as u64,dec_len);
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(ubase_fixed as u32,32);
        bitpack_vec.write((ubase_fixed>>32) as u32,32);
        bitpack_vec.write(t, 32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);

        // let duration1 = start1.elapsed();
        // println!("Time elapsed in dividing double function() is: {:?}", duration1);

        // let start1 = Instant::now();
        let mut remain = fixed_len;
        let mut bytec = 0;

        if remain<8{
            for i in fixed_vec{
                bitpack_vec.write_bits((i-base_fixed64) as u32, remain).unwrap();
            }
            remain = 0;
        }
        else {
            bytec+=1;
            remain -= 8;
            let mut fixed_u64 = Vec::new();
            let mut cur_u64 = 0u64;
            if remain>0{
                // let mut k = 0;
                fixed_u64 = fixed_vec.iter().map(|x|{
                    cur_u64 = (*x-base_fixed64) as u64;
                    bitpack_vec.write_byte((cur_u64>>remain) as u8);
                    cur_u64
                }).collect_vec();
            }
            else {
                fixed_u64 = fixed_vec.iter().map(|x|{
                    cur_u64 = (*x-base_fixed64) as u64;
                    bitpack_vec.write_byte((cur_u64) as u8);
                    cur_u64
                }).collect_vec();
            }
            println!("write the {}th byte of dec",bytec);

            while (remain>=8){
                bytec+=1;
                remain -= 8;
                if remain>0{
                    for d in &fixed_u64 {
                        bitpack_vec.write_byte((*d >>remain) as u8).unwrap();
                    }
                }
                else {
                    for d in &fixed_u64 {
                        bitpack_vec.write_byte(*d as u8).unwrap();
                    }
                }


                println!("write the {}th byte of dec",bytec);
            }
            if (remain>0){
                bitpack_vec.finish_write_byte();
                for d in fixed_u64 {
                    bitpack_vec.write_bits(d as u32, remain as usize).unwrap();
                }
                println!("write remaining {} bits of dec",remain);
            }
        }


        // println!("total number of dec is: {}", j);
        let vec = bitpack_vec.into_vec();

        // let duration1 = start1.elapsed();
        // println!("Time elapsed in writing double function() is: {:?}", duration1);

        let origin = t * mem::size_of::<T>() as u32;
        info!("original size:{}", origin);
        info!("compressed size:{}", vec.len());
        let ratio = vec.len() as f64 /origin as f64;
        print!("{}",ratio);
        vec
        //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }
pub fn byte_fixed_decode(&self, bytes: BorrowedBytes<'_>) -> Vec<f64>{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;

        let mut expected_datapoints:Vec<f64> = Vec::new();
        let mut fixed_vec:Vec<u64> = Vec::new();

        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen+ilen;
        let mut bytec = 0;
        let mut chunk;
        let mut f_cur = 0f64;

        if remain<8{
            for i in 0..len {
                cur = bitpack.read_bits(remain as usize).unwrap();
                expected_datapoints.push((base_int + cur as i64 ) as f64 / dec_scl);
            }
            remain=0
        }
        else {
            bytec+=1;
            remain -= 8;
            chunk = bitpack.read_n_byte(len as usize).unwrap();

            if remain == 0 {
                for &x in chunk {
                    expected_datapoints.push((base_int + x as i64) as f64 / dec_scl);
                }
            }
            else{
                // dec_vec.push((bitpack.read_byte().unwrap() as u32) << remain);
                // let mut k = 0;
                for x in chunk{
                    // if k<10{
                    //     println!("write {}th value with first byte {}",k,(*x))
                    // }
                    // k+=1;
                    fixed_vec.push(((*x) as u64)<<remain)
                }
            }
            println!("read the {}th byte of dec",bytec);

            while (remain>=8){
                bytec+=1;
                remain -= 8;
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                if remain == 0 {
                    // dec_vec=dec_vec.into_iter().map(|x| x|(bitpack.read_byte().unwrap() as u32)).collect();
                    // let mut iiter = int_vec.iter();
                    // let mut diter = dec_vec.iter();
                    // for cur_chunk in chunk.iter(){
                    //     expected_datapoints.push( *(iiter.next().unwrap()) as f64+ (((diter.next().unwrap())|((*cur_chunk) as u32)) as f64) / dec_scl);
                    // }

                    for (cur_fixed,cur_chunk) in fixed_vec.iter().zip(chunk.iter()){
                        expected_datapoints.push( (base_int + ((*cur_fixed)|((*cur_chunk) as u64)) as i64 ) as f64 / dec_scl);
                    }
                }
                else{
                    let mut it = chunk.into_iter();
                    fixed_vec=fixed_vec.into_iter().map(|x| x|((*(it.next().unwrap()) as u64)<<remain)).collect();
                }


                println!("read the {}th byte of dec",bytec);
            }
            // let duration = start.elapsed();
            // println!("Time elapsed in leading bytes: {:?}", duration);


            // let start5 = Instant::now();
            if (remain>0){
                bitpack.finish_read_byte();
                println!("read remaining {} bits of dec",remain);
                println!("length for fixed:{}", fixed_vec.len());
                for cur_fixed in fixed_vec.into_iter(){
                    f_cur = (base_int + ((cur_fixed)|(bitpack.read_bits( remain as usize).unwrap() as u64)) as i64) as f64 / dec_scl;
                    // todo: this is for value reconstruction
                    // format!("{:.2}", f_cur);
                    expected_datapoints.push( f_cur);
                }
            }
        }
        // for i in 0..10{
        //     println!("{}th item:{}",i,expected_datapoints.get(i).unwrap())
        // }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }
pub fn buff_decode_condition(&self, bytes: BorrowedBytes<'_>, iter: Iter<usize>) -> Vec<f64>{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;

        let mut expected_datapoints:Vec<f64> = Vec::new();
        let mut fixed_vec:Vec<u64> = Vec::new();

        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = (dlen+ilen) as usize;
        let mut bytec = 0;
        let mut pre = 0;
        let mut delta = 0;

        if remain<8{
            let mut it = iter.clone();
            pre = *it.next().unwrap();
            bitpack.skip(pre * remain);
            cur = bitpack.read_bits(remain).unwrap();
            expected_datapoints.push((base_int + cur as i64 ) as f64 / dec_scl);

            for &entry in it{
                delta = entry-pre;
                if delta != 1 {
                    bitpack.skip((delta-1) * remain);
                }
                cur = bitpack.read_bits(remain).unwrap();
                expected_datapoints.push((base_int + cur as i64 ) as f64 / dec_scl);
                pre = entry;
            }
            remain=0;
            return expected_datapoints;
        }
        else {
            bytec+=1;
            remain -= 8;

            if remain == 0 {

                let mut it = iter.clone();
                pre = *it.next().unwrap();
                bitpack.skip_n_byte(pre);
                cur = bitpack.read_byte().unwrap();
                expected_datapoints.push((base_int + cur as i64 ) as f64 / dec_scl);

                for &entry in it{
                    delta = entry-pre;
                    if delta != 1 {
                        bitpack.skip_n_byte((delta-1) );
                    }
                    cur = bitpack.read_byte().unwrap();
                    expected_datapoints.push((base_int + cur as i64 ) as f64 / dec_scl);
                    pre = entry;
                }
                if len - pre>1 {
                    bitpack.skip_n_byte((len - pre - 1) as usize);
                }
                return expected_datapoints;
            }
            else{
                let mut it = iter.clone();
                pre = *it.next().unwrap();
                bitpack.skip_n_byte((pre) );
                cur = bitpack.read_byte().unwrap();
                fixed_vec.push(((cur) as u64)<<remain);

                for &entry in it{
                    delta = entry-pre;
                    if delta != 1 {
                        bitpack.skip_n_byte((delta-1) );
                    }
                    cur = bitpack.read_byte().unwrap();
                    fixed_vec.push(((cur) as u64)<<remain);
                    pre = entry;
                }
                println!("len:{},pre:{}",len , pre);
                if len - pre>1 {
                    bitpack.skip_n_byte(((len - pre - 1) as usize));
                }
            }
            println!("read the {}th byte of dec",bytec);

            while (remain>=8){
                bytec+=1;
                remain -= 8;
                if remain == 0 {
                    let mut fixed_iter = fixed_vec.iter();
                    let mut it = iter.clone();
                    pre = *it.next().unwrap();
                    bitpack.skip_n_byte((pre) as usize);
                    cur = bitpack.read_byte().unwrap();
                    expected_datapoints.push((base_int + (cur as u64|*fixed_iter.next().unwrap()) as i64 ) as f64 / dec_scl);

                    for &entry in it{
                        delta = entry-pre;
                        if delta != 1 {
                            bitpack.skip_n_byte((delta-1) as usize);
                        }
                        cur = bitpack.read_byte().unwrap();
                        expected_datapoints.push((base_int + (cur as u64|*fixed_iter.next().unwrap()) as i64 ) as f64 / dec_scl);
                        pre = entry;
                    }

                }
                else{
                    let mut fixed_iter = fixed_vec.iter_mut();
                    let mut it = iter.clone();
                    pre = *it.next().unwrap();
                    bitpack.skip_n_byte((pre) as usize);
                    cur = bitpack.read_byte().unwrap();
                    let mut c_val = fixed_iter.next().unwrap();
                    *c_val = (*c_val)|(((cur) as u64)<<remain);

                    for &entry in it{
                        delta = entry-pre;
                        if delta != 1 {
                            bitpack.skip_n_byte((delta-1));
                        }
                        cur = bitpack.read_byte().unwrap();
                        c_val = fixed_iter.next().unwrap();
                        *c_val = (*c_val)|(((cur) as u64)<<remain);
                        pre = entry;
                    }
                    println!("len:{},pre:{}",len , pre);
                    if len - pre>1 {
                        bitpack.skip_n_byte((((len - pre - 1) as usize)) );
                    }

                }


                println!("read the {}th byte of dec",bytec);
            }
            // let duration = start.elapsed();
            // println!("Time elapsed in leading bytes: {:?}", duration);


            // let start5 = Instant::now();
            if (remain>0){
                bitpack.finish_read_byte();
                println!("read remaining {} bits of dec",remain);
                println!("length for fixed:{}", fixed_vec.len());

                let mut fixed_iter = fixed_vec.iter();
                let mut it = iter.clone();
                pre = *it.next().unwrap();
                bitpack.skip((pre * remain) as usize);
                cur = bitpack.read_bits(remain as usize).unwrap();
                expected_datapoints.push((base_int + ((*(fixed_iter.next().unwrap()) as u64)|((cur as u64)<< remain as u64) ) as i64 ) as f64 / dec_scl);

                for &entry in it{
                    delta = entry-pre;
                    if delta != 1 {
                        bitpack.skip(((delta-1) * remain));
                    }
                    cur = bitpack.read_bits(remain as usize).unwrap();
                    expected_datapoints.push((base_int + ((*(fixed_iter.next().unwrap()) as u64)|((cur as u64)<< remain as u64) )as i64 ) as f64 / dec_scl);
                    pre = entry;
                }

            }
        }
        // for i in 0..10{
        //     println!("{}th item:{}",i,expected_datapoints.get(i).unwrap())
        // }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }
}
