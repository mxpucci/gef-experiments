//! Thin C ABI over the official pco crate. All format/algorithm work stays upstream.
mod indexed;
use pco::data_types::Number;
use pco::standalone::{self, DecompressorItem, FileDecompressor};
use pco::{ChunkConfig, PagingSpec};
use std::{mem, panic, slice};

const OK: i32 = 0;
const INVALID: i32 = 1;
const CODEC_ERROR: i32 = 2;
const PANIC: i32 = 3;

fn guard(f: impl FnOnce() -> i32 + panic::UnwindSafe) -> i32 {
    panic::catch_unwind(f).unwrap_or(PANIC)
}

fn valid_slice<T>(ptr: *const T, len: usize) -> bool {
    len <= isize::MAX as usize / mem::size_of::<T>()
        && (len == 0 || (!ptr.is_null() && (ptr as usize) % mem::align_of::<T>() == 0))
}

// Empty C++ vectors may have null data(); Rust slices must never have null pointers.
unsafe fn input<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 { &[] } else { slice::from_raw_parts(ptr, len) }
}
unsafe fn output<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    if len == 0 { &mut [] } else { slice::from_raw_parts_mut(ptr, len) }
}

#[no_mangle]
pub unsafe extern "C" fn gef_pco_bound(n: usize, result: *mut usize) -> i32 {
    guard(|| {
        if result.is_null() || n > isize::MAX as usize / 16 { return INVALID; }
        match standalone::guarantee::file_size::<u64>(n, &PagingSpec::default()) {
            Ok(size) => { *result = size; OK },
            Err(_) => CODEC_ERROR,
        }
    })
}

unsafe fn compress<T: Number>(src: *const T, n: usize, dst: *mut u8,
    cap: usize, written: *mut usize) -> i32 {
    if !valid_slice(src, n) || !valid_slice(dst, cap) || written.is_null() { return INVALID; }
    let source = input(src, n);
    let target = output(dst, cap);
    match standalone::simple_compress_into(source, &ChunkConfig::default(), target) {
        Ok(tail) => { *written = cap - tail.len(); OK },
        Err(_) => CODEC_ERROR,
    }
}

unsafe fn decompress<T: Number>(src: *const u8, len: usize, dst: *mut T, n: usize) -> i32 {
    if !valid_slice(src, len) || !valid_slice(dst, n) { return INVALID; }
    let (file, mut source) = match FileDecompressor::new(input(src, len)) {
        Ok(pair) => pair,
        Err(_) => return CODEC_ERROR,
    };
    let mut target = output(dst, n);
    loop {
        match file.chunk_decompressor::<T, _>(source) {
            Ok(DecompressorItem::Chunk(mut chunk)) => {
                let count = chunk.n();
                if count > target.len() { return CODEC_ERROR; }
                let (chunk_target, rest) = target.split_at_mut(count);
                match chunk.read(chunk_target) {
                    Ok(progress) if progress.finished && progress.n_processed == count => {},
                    _ => return CODEC_ERROR,
                }
                source = chunk.into_src();
                target = rest;
            }
            // Require an exact file, exact output count, and no trailing garbage.
            Ok(DecompressorItem::EndOfData(rest)) =>
                return if target.is_empty() && rest.is_empty() { OK } else { CODEC_ERROR },
            Err(_) => return CODEC_ERROR,
        }
    }
}

macro_rules! export_codec {
    ($ty:ty, $compress:ident, $decompress:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $compress(src: *const $ty, n: usize, dst: *mut u8,
            cap: usize, written: *mut usize) -> i32 {
            guard(|| compress(src, n, dst, cap, written))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $decompress(src: *const u8, len: usize, dst: *mut $ty,
            n: usize) -> i32 {
            guard(|| decompress(src, len, dst, n))
        }
    }
}
export_codec!(f64, gef_pco_compress_f64, gef_pco_decompress_f64);
export_codec!(i64, gef_pco_compress_i64, gef_pco_decompress_i64);
