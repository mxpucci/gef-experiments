// FFI and serialization only. Compression/decompression algorithms are compiled
// directly from the pinned Apache-2.0 upstream source modules. cmake/Fauna.cmake
// applies upstream-f64.patch, which widens the upstream `Value = f32` alias to f64.
#![allow(dead_code)]
mod indexed;
include!(concat!(env!("OUT_DIR"), "/upstream.rs"));

use arrow::array::{Array, BinaryArray, UInt8Array};
use modelardb_types::types::{ErrorBound, TimestampArray, TimestampBuilder,
    UnivariateIdBuilder, Value, ValueArray, ValueBuilder};
use std::{cell::RefCell, ffi::CString, os::raw::c_char, panic::{catch_unwind, AssertUnwindSafe}};

// The C ABI exchanges `double`; the patched upstream alias must match it.
const _: () = assert!(std::mem::size_of::<Value>() == 8);

thread_local! { static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }

fn set_error(message: &str) {
    LAST_ERROR.with(|v| *v.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
}

#[no_mangle]
pub extern "C" fn gef_fauna_last_error() -> *const c_char {
    LAST_ERROR.with(|v| v.borrow().as_ptr())
}

fn put_u64(out: &mut Vec<u8>, v: u64) { out.extend_from_slice(&v.to_le_bytes()); }
fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u64(out, bytes.len() as u64); out.extend_from_slice(bytes);
}

struct Reader<'a> { bytes: &'a [u8], pos: usize }
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("frame overflow")?;
        let result = self.bytes.get(self.pos..end).ok_or("truncated frame")?;
        self.pos = end; Ok(result)
    }
    fn u64(&mut self) -> Result<u64, String> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn value(&mut self) -> Result<Value, String> {
        Ok(Value::from_le_bytes(self.take(std::mem::size_of::<Value>())?.try_into().unwrap()))
    }
    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let len = usize::try_from(self.u64()?).map_err(|_| "length overflow")?;
        self.take(len)
    }
}

fn encode(values: &[Value], macaque: bool) -> Result<Vec<u8>, String> {
    // Model fitting identifies -0 and +0 and treats NaNs as equal. Do not
    // promise bitwise preservation for those values with the Fauna model path.
    if !macaque && values.iter().any(|v| !v.is_finite() || v.to_bits() == (-0.0 as Value).to_bits()) {
        return Err("Fauna model fitting requires finite values without negative zero".into());
    }
    let mut out = Vec::new();
    out.extend_from_slice(if macaque { b"GFMAC002" } else { b"GFFAU002" });
    put_u64(&mut out, values.len() as u64);
    if macaque {
        if !values.is_empty() {
            let mut codec = models::gorilla::Gorilla::new(ErrorBound::Absolute(0.0));
            codec.compress_values(values);
            out.extend_from_slice(&codec.model().0);
        }
        return Ok(out);
    }
    let timestamps = TimestampArray::from_iter_values(0..values.len() as i64);
    let batch = compression::try_compress(0, ErrorBound::Absolute(0.0),
        &timestamps, &ValueArray::from(values.to_vec())).map_err(|e| e.to_string())?;
    put_u64(&mut out, batch.num_rows() as u64);
    for row in 0..batch.num_rows() {
        out.push(batch.column(1).as_any().downcast_ref::<UInt8Array>().unwrap().value(row));
        for col in [2, 3] {
            let t = batch.column(col).as_any().downcast_ref::<TimestampArray>().unwrap().value(row);
            put_u64(&mut out, t as u64);
        }
        for col in [5, 6] {
            let f = batch.column(col).as_any().downcast_ref::<ValueArray>().unwrap().value(row);
            out.extend_from_slice(&f.to_le_bytes());
        }
        for col in [4, 7, 8] {
            let bytes = batch.column(col).as_any().downcast_ref::<BinaryArray>().unwrap().value(row);
            put_bytes(&mut out, bytes);
        }
    }
    Ok(out)
}

fn decode(bytes: &[u8], count: usize, macaque: bool) -> Result<Vec<Value>, String> {
    let mut r = Reader { bytes, pos: 0 };
    if r.take(8)? != if macaque { b"GFMAC002" } else { b"GFFAU002" } { return Err("wrong codec/frame version".into()); }
    if r.u64()? != count as u64 { return Err("value count mismatch".into()); }
    let mut ids = UnivariateIdBuilder::new();
    let mut result = ValueBuilder::with_capacity(count);
    if macaque {
        if count > 0 {
            let timestamps: Vec<i64> = (0..count as i64).collect();
            models::gorilla::grid(0, &bytes[r.pos..], &mut ids, &timestamps, &mut result, None);
        } else if r.pos != bytes.len() { return Err("trailing bytes".into()); }
    } else {
        let segments = usize::try_from(r.u64()?).map_err(|_| "segment count overflow")?;
        if segments > count { return Err("too many segments".into()); }
        let mut timestamps = TimestampBuilder::with_capacity(count);
        for _ in 0..segments {
            let model = r.take(1)?[0];
            let first = r.u64()? as i64;
            let last = r.u64()? as i64;
            let min = r.value()?;
            let max = r.value()?;
            let times = r.bytes()?;
            let values = r.bytes()?;
            let residuals = r.bytes()?;
            if model > 2 || first != result.values_slice().len() as i64 || last < first || last >= count as i64 {
                return Err("invalid segment bounds/model".into());
            }
            models::grid(0, model, first, last, times, min, max, values, residuals,
                &mut ids, &mut timestamps, &mut result);
            if result.values_slice().len() != last as usize + 1 { return Err("segment length mismatch".into()); }
        }
        if r.pos != bytes.len() { return Err("trailing bytes".into()); }
    }
    if result.values_slice().len() != count { return Err("decoded count mismatch".into()); }
    Ok(result.values_slice().to_vec())
}

fn ffi_call(f: impl FnOnce() -> Result<(), String>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => { set_error(&e); 1 },
        Err(_) => { set_error("upstream Fauna/Macaque panic (invalid data or unsupported input)"); 2 }
    }
}

#[no_mangle]
pub unsafe extern "C" fn gef_fauna_compress(input: *const Value, count: usize, macaque: bool,
    output: *mut *mut u8, len: *mut usize) -> i32 {
    ffi_call(|| {
        if output.is_null() || len.is_null() || (count > 0 && input.is_null()) { return Err("null FFI pointer".into()); }
        if count > isize::MAX as usize / 8 { return Err("input too large".into()); }
        let values = if count == 0 { &[] } else { std::slice::from_raw_parts(input, count) };
        let bytes = encode(values, macaque)?.into_boxed_slice();
        *len = bytes.len(); *output = Box::into_raw(bytes) as *mut u8; Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn gef_fauna_decompress(input: *const u8, len: usize, count: usize,
    macaque: bool, output: *mut Value) -> i32 {
    ffi_call(|| {
        if input.is_null() || (count > 0 && output.is_null()) { return Err("null FFI pointer".into()); }
        if len > isize::MAX as usize || count > isize::MAX as usize / 8 { return Err("frame too large".into()); }
        let values = decode(std::slice::from_raw_parts(input, len), count, macaque)?;
        if count > 0 { std::ptr::copy_nonoverlapping(values.as_ptr(), output, count); }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn gef_fauna_free(bytes: *mut u8, len: usize) {
    if !bytes.is_null() { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(bytes, len))); }
}
