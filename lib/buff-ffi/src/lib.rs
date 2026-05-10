//! BUFF compressor C FFI.
//!
//! Encode/decode logic vendored verbatim from
//! https://github.com/Tranway1/buff (MIT licence).
//!
//! Only `buff_simd256_encode` / `buff_simd256_decode` and their direct
//! helpers are included.  The database infrastructure (RocksDB, Parquet,
//! Tokio, CRoaring) that lives in the same repo is not needed here.
//!
//! Wire format (prepended by this crate):
//!   [0..8]  n     (u64 LE) — number of f64 values
//!   [8..16] scale (i64 LE) — 10^decimals used during encode
//!   [16..]  BUFF compressed stream (self-describing header inside)

const FFI_HEADER: usize = 16;

// ── Vendored BUFF compression ─────────────────────────────────────────────
//
// Pure scalar; builds on any 64-bit target.  Upstream's "simd256" name
// refers to a separate AVX-256 code path we did not vendor.

mod inner {
    use super::FFI_HEADER;
    use std::mem;

    // ── PRECISION_MAP ─────────────────────────────────────────────────────
    // From buff/database/src/compress/mod.rs.
    // Maps prec = log10(scale) → decimal bit-length used in fixed-point repr.
    fn dec_len_for_prec(prec: i32) -> Option<u64> {
        match prec {
            1  => Some(5),  2  => Some(8),  3  => Some(11), 4  => Some(15),
            5  => Some(18), 6  => Some(21), 7  => Some(25), 8  => Some(28),
            9  => Some(31), 10 => Some(35), 11 => Some(38), 12 => Some(50),
            13 => Some(10), 14 => Some(10), 15 => Some(10),
            _  => None,
        }
    }

    // ── PrecisionBound ────────────────────────────────────────────────────
    // From buff/database/src/methods/prec_double.rs.
    // Only the fields and methods used by encode/decode are kept.

    const EXP_MASK:  u64 = 0x7FF0_0000_0000_0000;
    const FIRST_ONE: u64 = 0x8000_0000_0000_0000;

    struct PrecisionBound {
        precision_exp:  i32,
        decimal_length: u64,
    }

    impl PrecisionBound {
        fn new(precision: f64) -> Self {
            let xu = unsafe { mem::transmute::<f64, u64>(precision) };
            PrecisionBound {
                precision_exp:  ((xu & EXP_MASK) >> 52) as i32 - 1023,
                decimal_length: 0,
            }
        }

        #[inline]
        fn set_length(&mut self, _ilen: u64, dlen: u64) {
            self.decimal_length = dlen;
        }

        #[inline]
        fn fetch_fixed_aligned(&self, bd: f64) -> i64 {
            let bdu  = unsafe { mem::transmute::<f64, u64>(bd) };
            let exp  = ((bdu & EXP_MASK) >> 52) as i32 - 1023;
            let sign = bdu & FIRST_ONE;
            let fixed = if exp < self.precision_exp {
                0u64
            } else {
                let f = ((bdu << 11) | FIRST_ONE)
                    >> (63 - exp - self.decimal_length as i32) as u64;
                if sign != 0 { !(f.wrapping_sub(1)) } else { f }
            };
            unsafe { mem::transmute::<u64, i64>(fixed) }
        }
    }

    // From buff/database/src/methods/prec_double.rs.
    fn get_precision_bound(prec: i32) -> f64 {
        let mut s = "0.".to_string();
        for _ in 0..prec { s.push('0'); }
        s.push_str("49");
        s.parse().unwrap()
    }

    // ── BitPack ───────────────────────────────────────────────────────────
    // From buff/database/src/methods/bit_packing.rs.
    // Supports writing/reading individual bits and bytes into a byte buffer.

    const MAX_BITS:  usize = 32;
    const BYTE_BITS: usize = 8;

    struct BitPack<B> {
        buff:   B,
        cursor: usize,
        bits:   usize,
    }

    impl<B> BitPack<B> {
        fn new(buff: B) -> Self { BitPack { buff, cursor: 0, bits: 0 } }
        #[inline] fn sum_bits(&self) -> usize { self.cursor * BYTE_BITS + self.bits }
    }

    // — Vec<u8> (writer) —
    impl BitPack<Vec<u8>> {
        fn with_capacity(cap: usize) -> Self { Self::new(Vec::with_capacity(cap)) }

        fn write(&mut self, mut value: u32, mut bits: usize) -> Result<(), usize> {
            // grow buffer
            let needed = self.sum_bits() + bits;
            let len = self.buff.len();
            if len * BYTE_BITS < needed {
                self.buff.resize(len + (needed - len * BYTE_BITS + BYTE_BITS - 1) / BYTE_BITS, 0);
            }
            if bits < MAX_BITS { value &= (1u32 << bits) - 1; }
            loop {
                let bits_left = BYTE_BITS - self.bits;
                if bits <= bits_left {
                    self.buff[self.cursor] |= (value as u8) << self.bits;
                    self.bits += bits;
                    if self.bits >= BYTE_BITS { self.cursor += 1; self.bits = 0; }
                    break;
                }
                let bb = value & ((1u32 << bits_left) - 1);
                self.buff[self.cursor] |= (bb as u8) << self.bits;
                self.cursor += 1;
                self.bits   = 0;
                value >>= bits_left;
                bits  -= bits_left;
            }
            Ok(())
        }

        #[inline]
        fn write_bits(&mut self, value: u32, bits: usize) -> Result<(), usize> {
            let masked = if bits < MAX_BITS { value & ((1u32 << bits) - 1) } else { value };
            self.write(masked, bits)
        }

        #[inline]
        fn write_byte(&mut self, value: u8) -> Result<(), usize> {
            self.buff.push(value);
            Ok(())
        }

        fn finish_write_byte(&mut self) {
            let len = self.buff.len();
            self.buff.resize(len + 1, 0);
            self.bits   = 0;
            self.cursor = len;
        }

        fn into_vec(self) -> Vec<u8> { self.buff }
    }

    // — &[u8] (reader) —
    impl<'a> BitPack<&'a [u8]> {
        fn read(&mut self, mut bits: usize) -> Result<u32, usize> {
            if bits > MAX_BITS || self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
                return Err(bits);
            }
            let mut out_shift = 0usize;
            let mut output    = 0u32;
            loop {
                let byte_left = BYTE_BITS - self.bits;
                if bits <= byte_left {
                    let bb = (self.buff[self.cursor] as u32 >> self.bits) & ((1u32 << bits) - 1);
                    output |= bb << out_shift;
                    self.bits += bits;
                    break;
                }
                let bb = (self.buff[self.cursor] as u32 >> self.bits) & ((1u32 << byte_left) - 1);
                output    |= bb << out_shift;
                self.bits += byte_left;
                out_shift += byte_left;
                bits      -= byte_left;
                if self.bits >= BYTE_BITS { self.cursor += 1; self.bits -= BYTE_BITS; }
            }
            Ok(output)
        }

        fn read_bits(&mut self, mut bits: usize) -> Result<u8, usize> {
            if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
                return Err(bits);
            }
            let mut out_shift = 0usize;
            let mut output    = 0u32;
            loop {
                let byte_left = BYTE_BITS - self.bits;
                if bits <= byte_left {
                    let bb = (self.buff[self.cursor] as u32 >> self.bits) & ((1u32 << bits) - 1);
                    output |= bb << out_shift;
                    self.bits += bits;
                    break;
                }
                let bb = (self.buff[self.cursor] as u32 >> self.bits) & ((1u32 << byte_left) - 1);
                output    |= bb << out_shift;
                self.bits += byte_left;
                out_shift += byte_left;
                bits      -= byte_left;
                if self.bits >= BYTE_BITS { self.cursor += 1; self.bits -= BYTE_BITS; }
            }
            Ok(output as u8)
        }

        fn read_n_byte(&mut self, n: usize) -> Result<&[u8], usize> {
            self.cursor += 1;
            let end = self.cursor + n;
            if end > self.buff.len() { return Err(n); }
            let out = &self.buff[self.cursor..end];
            self.cursor += n - 1;
            Ok(out)
        }

        #[inline]
        fn finish_read_byte(&mut self) {
            self.cursor += 1;
            self.bits   = 0;
        }
    }

    // ── flip ──────────────────────────────────────────────────────────────
    // From buff/database/src/compress/buff_slice.rs.
    #[inline]
    fn flip(x: u8) -> u8 { x ^ 0x80 }

    // ── Scale validation ──────────────────────────────────────────────────
    fn valid_prec(scale: i64) -> Option<i32> {
        if scale <= 1 { return None; }
        let prec = (scale as f64).log10().round() as i32;
        dec_len_for_prec(prec)?;
        Some(prec)
    }

    // ── buff_simd256_encode ───────────────────────────────────────────────
    // Vendored from buff/database/src/compress/buff_simd.rs.
    // Adapted to take &[f64] directly instead of Segment<T>.
    fn buff_encode(data: &[f64], scale: i64) -> Option<Vec<u8>> {
        let prec    = valid_prec(scale)?;
        let dec_len = dec_len_for_prec(prec)?;

        let prec_delta = get_precision_bound(prec);
        let mut bound  = PrecisionBound::new(prec_delta);
        bound.set_length(0, dec_len);

        let mut fixed_vec: Vec<i64> = Vec::with_capacity(data.len());
        let mut min = i64::MAX;
        let mut max = i64::MIN;

        for &val in data {
            let fixed = bound.fetch_fixed_aligned(val);
            if fixed < min { min = fixed; }
            if fixed > max { max = fixed; }
            fixed_vec.push(fixed);
        }

        let t     = data.len() as u32;
        let delta = max.wrapping_sub(min);

        // Two minimal fixes over upstream BUFF (no algorithm change, no
        // format change):
        //
        //  (1) Bit-length instead of log2().ceil().  Upstream's
        //      `(delta as f64).log2().ceil()` under-counts by 1 when delta
        //      is an exact power of 2 (e.g. delta = 256 gives 8, but
        //      encoding 0..=256 needs 9 bits).  The top bit is silently
        //      truncated → decoded value is 2^dec_len smaller than encoded.
        //      `64 - leading_zeros(delta)` gives the correct bit-length and
        //      matches log2().ceil() exactly on every non-power-of-2 input.
        //  (2) Pad fixed_len up to dec_len.  Upstream's
        //      `let ilen = fixed_len - dec_len` underflows u64 when delta
        //      is too narrow.  The clamp encodes narrow blocks with
        //      leading-zero bits at the same precision bound (1/2^dec_len)
        //      as wide blocks.
        //
        // Combined, the encoder is byte-identical to upstream on every
        // block where (a) delta > 0, (b) delta is not a power of 2, and
        // (c) bit_length(delta) >= dec_len — i.e., every block upstream
        // would have encoded correctly.
        let cal_int_length: usize = if delta == 0 {
            0
        } else {
            64 - (delta as u64).leading_zeros() as usize
        };
        let fixed_len = std::cmp::max(cal_int_length, dec_len as usize);
        let ilen = fixed_len - dec_len as usize;  // safe: fixed_len ≥ dec_len
        let dlen = dec_len as usize;

        bound.set_length(ilen as u64, dlen as u64);

        let base_fixed64  = min;
        let ubase_fixed   = unsafe { mem::transmute::<i64, u64>(base_fixed64) };

        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(ubase_fixed as u32,       32);
        bitpack_vec.write((ubase_fixed >> 32) as u32, 32);
        bitpack_vec.write(t,       32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);

        let mut remain = fixed_len;

        if remain < 8 {
            for &i in &fixed_vec {
                bitpack_vec.write_bits((i - base_fixed64) as u32, remain).unwrap();
            }
        } else {
            remain -= 8;
            let mut fixed_u64: Vec<u64>;

            if remain > 0 {
                fixed_u64 = fixed_vec.iter().map(|&x| {
                    let cur = (x - base_fixed64) as u64;
                    bitpack_vec.write_byte(flip((cur >> remain) as u8)).unwrap();
                    cur
                }).collect();
            } else {
                fixed_u64 = fixed_vec.iter().map(|&x| {
                    let cur = (x - base_fixed64) as u64;
                    bitpack_vec.write_byte(flip(cur as u8)).unwrap();
                    cur
                }).collect();
            }

            while remain >= 8 {
                remain -= 8;
                if remain > 0 {
                    for &d in &fixed_u64 {
                        bitpack_vec.write_byte(flip((d >> remain) as u8)).unwrap();
                    }
                } else {
                    for &d in &fixed_u64 {
                        bitpack_vec.write_byte(flip(d as u8)).unwrap();
                    }
                }
            }

            if remain > 0 {
                bitpack_vec.finish_write_byte();
                for d in fixed_u64 {
                    bitpack_vec.write_bits(d as u32, remain).unwrap();
                }
            }
        }

        Some(bitpack_vec.into_vec())
    }

    // ── buff_simd256_decode ───────────────────────────────────────────────
    // Vendored from buff/database/src/compress/buff_simd.rs.
    fn buff_decode(bytes: Vec<u8>, scale: i64) -> Option<Vec<f64>> {
        let prec       = valid_prec(scale)?;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound   = PrecisionBound::new(prec_delta);

        let lower    = bitpack.read(32).ok()?;
        let higher   = bitpack.read(32).ok()?;
        let ubase    = (lower as u64) | ((higher as u64) << 32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase) };

        let len  = bitpack.read(32).ok()? as usize;
        let ilen = bitpack.read(32).ok()?;
        let dlen = bitpack.read(32).ok()?;
        bound.set_length(ilen as u64, dlen as u64);

        let dec_scl = 2.0f64.powi(dlen as i32);
        let mut remain = dlen + ilen;
        let mut expected: Vec<f64>  = Vec::with_capacity(len);
        let mut fixed_vec: Vec<u64> = Vec::with_capacity(len);

        if remain < 8 {
            for _ in 0..len {
                let cur = bitpack.read_bits(remain as usize).ok()? as u64;
                expected.push((base_int + cur as i64) as f64 / dec_scl);
            }
        } else {
            remain -= 8;
            let chunk = bitpack.read_n_byte(len).ok()?;

            if remain == 0 {
                for &x in chunk {
                    expected.push((base_int + flip(x) as i64) as f64 / dec_scl);
                }
            } else {
                for &x in chunk {
                    fixed_vec.push((flip(x) as u64) << remain);
                }
            }

            while remain >= 8 {
                remain -= 8;
                let chunk = bitpack.read_n_byte(len).ok()?;
                if remain == 0 {
                    for (cur_fixed, &cur_chunk) in fixed_vec.iter().zip(chunk.iter()) {
                        expected.push(
                            (base_int + (*cur_fixed | flip(cur_chunk) as u64) as i64) as f64
                                / dec_scl,
                        );
                    }
                } else {
                    let mut it = chunk.iter();
                    fixed_vec = fixed_vec
                        .into_iter()
                        .map(|x| x | ((flip(*it.next().unwrap()) as u64) << remain))
                        .collect();
                }
            }

            if remain > 0 {
                bitpack.finish_read_byte();
                for cur_fixed in fixed_vec {
                    let cur = bitpack.read_bits(remain as usize).ok()? as u64;
                    expected.push((base_int + (cur_fixed | cur) as i64) as f64 / dec_scl);
                }
            }
        }

        Some(expected)
    }

    // ── Single-value extraction (random access) ───────────────────────────
    //
    // BUFF's byte-major layout makes O(1)-per-query random access natural:
    // value `i` is reconstructed by reading byte `i` from each byte-plane
    // (one byte per plane in the block) plus a few trailing bits (if any).
    // Cost: ~ceil(fixed_len/8) byte reads, no full-block decode, no alloc.
    //
    // Mirrors `buff_decode`'s two regimes:
    //   • fixed_len < 8: the body is bit-packed `fixed_len` bits per value
    //     starting at byte 20.
    //   • fixed_len >= 8: byte planes [byte 20, +len, +2·len, …] each
    //     `len` bytes long, followed (if fixed_len % 8 != 0) by the
    //     trailing bits packed continuously starting at byte 20+K·len.
    fn buff_extract(bytes: &[u8], scale: i64, idx: usize) -> Option<f64> {
        let _prec = valid_prec(scale)?;
        if bytes.len() < 20 { return None; }

        let read_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]])
        };
        let lower    = read_u32(0);
        let higher   = read_u32(4);
        let ubase    = (lower as u64) | ((higher as u64) << 32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase) };
        let len      = read_u32(8) as usize;
        let ilen     = read_u32(12);
        let dlen     = read_u32(16);

        if idx >= len { return None; }

        let fixed_len = (ilen + dlen) as usize;
        let dec_scl   = 2.0f64.powi(dlen as i32);
        let body      = 20usize;

        let cur: u64 = if fixed_len == 0 {
            0
        } else if fixed_len < 8 {
            let bit_offset = idx * fixed_len;
            read_bits_at(bytes, body + bit_offset / 8, bit_offset % 8, fixed_len)?
        } else {
            let num_planes = fixed_len / 8;
            let trailing   = fixed_len % 8;
            let mut acc: u64 = 0;
            for k in 0..num_planes {
                let pos = body + k * len + idx;
                if pos >= bytes.len() { return None; }
                let unflipped = bytes[pos] ^ 0x80;  // inverse of flip()
                let shift = trailing + (num_planes - 1 - k) * 8;
                acc |= (unflipped as u64) << shift;
            }
            if trailing > 0 {
                let trailing_byte_start = body + num_planes * len;
                let bit_offset = idx * trailing;
                let byte_idx = trailing_byte_start + bit_offset / 8;
                let bit_idx  = bit_offset % 8;
                acc |= read_bits_at(bytes, byte_idx, bit_idx, trailing)?;
            }
            acc
        };

        Some((base_int + cur as i64) as f64 / dec_scl)
    }

    fn read_bits_at(bytes: &[u8], byte_idx: usize, bit_idx: usize, n_bits: usize) -> Option<u64> {
        if n_bits == 0 { return Some(0); }
        if byte_idx >= bytes.len() { return None; }
        let mut result: u64 = 0;
        let mut taken = 0;
        let mut byte = byte_idx;
        let mut bit  = bit_idx;
        while taken < n_bits {
            if byte >= bytes.len() { return None; }
            let avail = 8 - bit;
            let take  = std::cmp::min(n_bits - taken, avail);
            let mask  = if take == 64 { !0u64 } else { (1u64 << take) - 1 };
            let chunk = ((bytes[byte] as u64) >> bit) & mask;
            result |= chunk << taken;
            taken += take;
            byte  += 1;
            bit    = 0;
        }
        Some(result)
    }

    // ── Public helpers called from C ABI ──────────────────────────────────

    pub fn max_size(n: usize) -> usize {
        FFI_HEADER + n * 8 + 128
    }

    pub fn compress(data: &[f64], out: &mut [u8], scale: i64) -> Option<usize> {
        let compressed = buff_encode(data, scale)?;
        let total = FFI_HEADER + compressed.len();
        if out.len() < total { return None; }
        out[0..8].copy_from_slice(&(data.len() as u64).to_le_bytes());
        out[8..16].copy_from_slice(&scale.to_le_bytes());
        out[FFI_HEADER..total].copy_from_slice(&compressed);
        Some(total)
    }

    pub fn decompress(input: &[u8], out: &mut [f64]) -> bool {
        if input.len() < FFI_HEADER { return false; }
        let n     = u64::from_le_bytes(input[0..8].try_into().unwrap()) as usize;
        let scale = i64::from_le_bytes(input[8..16].try_into().unwrap());
        if n != out.len() { return false; }
        let buff_bytes = input[FFI_HEADER..].to_vec();
        match buff_decode(buff_bytes, scale) {
            Some(dec) if dec.len() == n => { out.copy_from_slice(&dec); true }
            _ => false,
        }
    }

    pub fn extract(input: &[u8], idx: usize) -> Option<f64> {
        if input.len() < FFI_HEADER { return None; }
        let scale = i64::from_le_bytes(input[8..16].try_into().unwrap());
        buff_extract(&input[FFI_HEADER..], scale, idx)
    }
}

// ── Public C ABI ──────────────────────────────────────────────────────────

/// Upper bound on compressed output size (bytes) for `n` f64 values.
#[no_mangle]
pub extern "C" fn buff_max_compressed_size(n: usize) -> usize {
    inner::max_size(n)
}

/// Compress `n` f64 values.
///
/// `scale` = 10^decimals (must be ≥ 10, i.e. at least 1 decimal place).
/// `output` must be at least `buff_max_compressed_size(n)` bytes.
///
/// Returns bytes written, or -1 on error.
#[no_mangle]
pub extern "C" fn buff_compress_f64(
    input: *const f64,
    n: usize,
    output: *mut u8,
    output_capacity: usize,
    scale: i64,
) -> i64 {
    if input.is_null() || output.is_null() || n == 0 { return -1; }
    let data = unsafe { std::slice::from_raw_parts(input, n) };
    let out  = unsafe { std::slice::from_raw_parts_mut(output, output_capacity) };
    match inner::compress(data, out, scale) {
        Some(len) => len as i64,
        None      => -1,
    }
}

/// Decompress `n` f64 values from BUFF-compressed data.
///
/// Returns 0 on success, -1 on error / unsupported platform.
#[no_mangle]
pub extern "C" fn buff_decompress_f64(
    input: *const u8,
    input_len: usize,
    output: *mut f64,
    n: usize,
) -> i32 {
    if input.is_null() || output.is_null() || n == 0 { return -1; }
    let data = unsafe { std::slice::from_raw_parts(input, input_len) };
    let out  = unsafe { std::slice::from_raw_parts_mut(output, n) };
    if inner::decompress(data, out) { 0 } else { -1 }
}

/// Extract a single f64 value at position `idx` from a BUFF-compressed block
/// without decoding the whole block.  Reads ~ceil(fixed_len/8) bytes (one byte
/// per byte-plane) plus a few trailing bits.  No allocation.
///
/// Returns 0 on success (writes the value to *output), -1 on error.
#[no_mangle]
pub extern "C" fn buff_extract_f64(
    input: *const u8,
    input_len: usize,
    idx: usize,
    output: *mut f64,
) -> i32 {
    if input.is_null() || output.is_null() { return -1; }
    let data = unsafe { std::slice::from_raw_parts(input, input_len) };
    match inner::extract(data, idx) {
        Some(v) => { unsafe { *output = v; } 0 }
        None    => -1,
    }
}

#[cfg(test)]
mod patch_tests {
    use super::*;

    fn roundtrip(values: &[f64], scale: i64) -> Vec<f64> {
        let cap = buff_max_compressed_size(values.len());
        let mut buf = vec![0u8; cap];
        let written = buff_compress_f64(values.as_ptr(), values.len(),
                                        buf.as_mut_ptr(), buf.len(), scale);
        assert!(written > 0, "compression failed");
        let mut out = vec![0.0f64; values.len()];
        let ok = buff_decompress_f64(buf.as_ptr(), written as usize,
                                     out.as_mut_ptr(), out.len());
        assert_eq!(ok, 0);
        out
    }

    #[test]
    fn us_bin_narrow_range() {
        // Mirrors US.bin: prec=2 (decimals=2), values with raw_int range ~8.
        let raw = [2584i64, 2584, 2584, 2586, 2584, 2581, 2583, 2584,
                   2584, 2585, 2584, 2586, 2584, 2578];
        let v: Vec<f64> = raw.iter().map(|&x| x as f64 / 100.0).collect();
        let out = roundtrip(&v, 100);
        // Verify within BUFF's 0.5/scale tolerance.
        for (a, b) in v.iter().zip(out.iter()) {
            assert!((a - b).abs() <= 0.5 / 100.0, "expected {a} got {b}");
        }
    }

    #[test]
    fn delta_zero_block() {
        // Constant block: previously rejected, now pads to dec_len bits.
        let v = [3.14_f64; 100];
        let out = roundtrip(&v, 100);
        for o in &out { assert!((o - 3.14).abs() <= 0.5 / 100.0); }
    }

    #[test]
    fn power_of_two_delta() {
        // Block where fetch_fixed_aligned delta = 2^8 exactly: upstream
        // log2().ceil() gives 8 bits, but encoding 0..=256 needs 9.  The
        // off-by-one truncated the top bit → decoded value was 2^dec_len
        // smaller than encoded (manifested as "off by 1.0" on the AP, ECG,
        // GE, IT, US datasets).
        let v = [17.0_f64, 17.5, 17.98, 18.5, 19.0];  // raw_int range chosen
        // so fetch_fixed_aligned spans an exact power of 2.  Mirrors the
        // pattern observed in production: expected 17.98, got 16.97...
        let out = roundtrip(&v, 100);
        for (a, b) in v.iter().zip(out.iter()) {
            assert!((a - b).abs() <= 0.5 / 100.0,
                    "expected {a} got {b} (delta {})", a - b);
        }
    }

    fn extract_via_ffi(values: &[f64], scale: i64) -> Vec<f64> {
        let cap = buff_max_compressed_size(values.len());
        let mut buf = vec![0u8; cap];
        let written = buff_compress_f64(values.as_ptr(), values.len(),
                                        buf.as_mut_ptr(), buf.len(), scale);
        assert!(written > 0, "compression failed");
        let mut out = vec![0.0f64; values.len()];
        for i in 0..values.len() {
            let mut v = 0.0f64;
            let ok = buff_extract_f64(buf.as_ptr(), written as usize, i, &mut v as *mut f64);
            assert_eq!(ok, 0, "extract failed at idx {i}");
            out[i] = v;
        }
        out
    }

    #[test]
    fn extract_matches_decompress() {
        // All three layout regimes:
        //   (a) fixed_len < 8: narrow data, prec=1, dec_len=5
        //   (b) fixed_len >= 8 with trailing bits: prec=2, dec_len=8, varied data
        //   (c) fixed_len multiple of 8: prec=2, dec_len=8, exactly 16-bit deltas
        let cases: &[(&[f64], i64)] = &[
            (&[3.1, 3.2, 3.3, 3.4, 3.5, 3.6], 10),
            (&[25.78, 26.5, 27.2, 28.1, 29.0, 30.5], 100),
            (&[100.5, 200.5, 300.5, 400.5, 500.5], 100),
            (&[17.0, 17.5, 17.98, 18.5, 19.0], 100),  // power-of-2 delta case
        ];
        for (vals, scale) in cases {
            let extracted = extract_via_ffi(vals, *scale);
            let decompressed = roundtrip(vals, *scale);
            assert_eq!(extracted.len(), decompressed.len());
            for (a, b) in extracted.iter().zip(decompressed.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(),
                    "extract vs decompress mismatch on {:?} (scale={}): {a} vs {b}",
                    vals, scale);
            }
        }
    }
}
