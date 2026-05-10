//! C FFI wrapper around the BUFF Rust implementation (https://github.com/Tranway1/buff).
//!
//! On x86_64 this calls the original `SplitBDDoubleCompress::buff_simd256_encode` /
//! `buff_simd256_decode` functions from the buff crate.  On non-x86_64 all entry points
//! return an error value, so the benchmark simply skips BUFF for that platform.
//!
//! The wire format prepends a 16-byte FFI header to buff's own compressed stream:
//!
//!   [0..8]  n     (u64 LE) — number of values
//!   [8..16] scale (i64 LE) — power-of-10 multiplier used during encode
//!   [16..]  buff's compressed output (self-describing, owns ilen/dlen in its header)
//!
//! The scale must satisfy `1 ≤ log10(scale) ≤ 15`; pass `10^decimals` for lossless
//! compression of fixed-precision datasets.  Datasets with 0 decimal places (scale=1)
//! cannot be handled by BUFF and will return -1.

const FFI_HEADER: usize = 16; // 8 bytes n + 8 bytes scale

// ── x86_64: call the real BUFF implementation ─────────────────────────────

#[cfg(target_arch = "x86_64")]
mod inner {
    use super::FFI_HEADER;
    use buff::compress::split_double::SplitBDDoubleCompress;
    use buff::segment::Segment;
    use std::time::SystemTime;

    /// Validate that `scale` is a power of 10 in [10, 10^15].
    /// buff's PRECISION_MAP only has entries for prec = log10(scale) in 1..=15.
    fn valid_prec(scale: i64) -> Option<i32> {
        if scale <= 1 {
            return None;
        }
        let prec = (scale as f64).log10().round() as i32;
        if prec < 1 || prec > 15 {
            return None;
        }
        Some(prec)
    }

    pub fn max_size(n: usize) -> usize {
        // buff header (12 B) + up to 8 bytes/value + generous padding
        FFI_HEADER + n * 8 + 128
    }

    pub fn compress(data: &[f64], out: &mut [u8], scale: i64) -> Option<usize> {
        valid_prec(scale)?;

        let mut seg = Segment::new(
            None,
            SystemTime::now(),
            0,
            data.to_vec(),
            None,
            None,
        );

        let comp = SplitBDDoubleCompress::new(10, 10, scale as usize);
        let compressed = comp.buff_simd256_encode(&mut seg);

        let total = FFI_HEADER + compressed.len();
        if out.len() < total {
            return None;
        }

        out[0..8].copy_from_slice(&(data.len() as u64).to_le_bytes());
        out[8..16].copy_from_slice(&scale.to_le_bytes());
        out[FFI_HEADER..total].copy_from_slice(&compressed);

        Some(total)
    }

    pub fn decompress(input: &[u8], out: &mut [f64]) -> bool {
        if input.len() < FFI_HEADER {
            return false;
        }
        let n     = u64::from_le_bytes(input[0..8].try_into().unwrap())   as usize;
        let scale = i64::from_le_bytes(input[8..16].try_into().unwrap());

        if valid_prec(scale).is_none() || n != out.len() {
            return false;
        }

        let buff_bytes: Vec<u8> = input[FFI_HEADER..].to_vec();
        let comp = SplitBDDoubleCompress::new(10, 10, scale as usize);
        let decompressed = comp.buff_simd256_decode(buff_bytes);

        if decompressed.len() != n {
            return false;
        }
        out.copy_from_slice(&decompressed);
        true
    }
}

// ── Public C ABI ──────────────────────────────────────────────────────────

/// Upper bound on compressed output size (bytes) for `n` f64 values.
#[no_mangle]
pub extern "C" fn buff_max_compressed_size(n: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    { inner::max_size(n) }
    #[cfg(not(target_arch = "x86_64"))]
    { let _ = n; 0 }
}

/// Compress `n` f64 values.
///
/// `scale` = 10^decimals for lossless fixed-precision data (must be ≥ 10).
/// `output` must be at least `buff_max_compressed_size(n)` bytes.
///
/// Returns bytes written on success, or -1 on error / unsupported platform.
#[no_mangle]
pub extern "C" fn buff_compress_f64(
    input: *const f64,
    n: usize,
    output: *mut u8,
    output_capacity: usize,
    scale: i64,
) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if input.is_null() || output.is_null() || n == 0 {
            return -1;
        }
        let data = unsafe { std::slice::from_raw_parts(input, n) };
        let out  = unsafe { std::slice::from_raw_parts_mut(output, output_capacity) };
        match inner::compress(data, out, scale) {
            Some(len) => len as i64,
            None      => -1,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    { let _ = (input, n, output, output_capacity, scale); -1 }
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
    #[cfg(target_arch = "x86_64")]
    {
        if input.is_null() || output.is_null() || n == 0 {
            return -1;
        }
        let data = unsafe { std::slice::from_raw_parts(input, input_len) };
        let out  = unsafe { std::slice::from_raw_parts_mut(output, n) };
        if inner::decompress(data, out) { 0 } else { -1 }
    }
    #[cfg(not(target_arch = "x86_64"))]
    { let _ = (input, input_len, output, n); -1 }
}
