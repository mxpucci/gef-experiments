//! Native Fauna model/residual segments with a counted positional directory.
//! Model and residual decoding calls the authors' original routines.
use super::*;
use crate::types::CompressedSegmentBuilder;
use std::ffi::c_void;

const MAGIC: &[u8; 8] = b"GFFIDX02";
const HEADER: usize = 32;
const ENTRY: usize = 32;
type Result<T> = std::result::Result<T, String>;
fn put(bytes: &mut [u8], at: usize, value: usize) {
    bytes[at..at + 8].copy_from_slice(&(value as u64).to_le_bytes());
}
fn word(bytes: &[u8], at: usize) -> Result<usize> {
    let b = bytes
        .get(at..at.checked_add(8).ok_or("index overflow")?)
        .ok_or("truncated index")?;
    usize::try_from(u64::from_le_bytes(b.try_into().unwrap())).map_err(|_| "index overflow".into())
}
struct Segment<'a> {
    model: u8,
    first: usize,
    last: usize,
    min: Value,
    max: Value,
    times: &'a [u8],
    values: &'a [u8],
    residuals: &'a [u8],
}
fn segment(bytes: &[u8]) -> Result<Segment<'_>> {
    let mut r = Reader { bytes, pos: 0 };
    let model = r.take(1)?[0];
    let first = r.u64()? as usize;
    let last = r.u64()? as usize;
    let min = r.value()?;
    let max = r.value()?;
    let times = r.bytes()?;
    let values = r.bytes()?;
    let residuals = r.bytes()?;
    if r.pos != bytes.len() || model > 2 || last < first || last > i64::MAX as usize {
        return Err("invalid native segment".into());
    }
    let residual_n = residuals.last().copied().unwrap_or(0) as usize;
    if residual_n >= last - first + 1 {
        return Err("segment without model values".into());
    }
    Ok(Segment {
        model,
        first,
        last,
        min,
        max,
        times,
        values,
        residuals,
    })
}
pub struct Indexed {
    bytes: Vec<u8>,
    n: usize,
    segments: usize,
}
impl Indexed {
    fn at(&self, i: usize) -> usize {
        HEADER + ENTRY * i
    }
    fn get(&self, i: usize) -> Result<Segment<'_>> {
        let at = self.at(i);
        let off = word(&self.bytes, at + 16)?;
        let len = word(&self.bytes, at + 24)?;
        segment(
            self.bytes
                .get(off..off.checked_add(len).ok_or("offset overflow")?)
                .ok_or("invalid segment bounds")?,
        )
    }
    fn new(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < HEADER || &bytes[..8] != MAGIC {
            return Err("wrong indexed Fauna format".into());
        }
        let n = word(&bytes, 8)?;
        let segments = word(&bytes, 16)?;
        let batch = word(&bytes, 24)?;
        if batch == 0 || batch > i64::MAX as usize || segments > (bytes.len() - HEADER) / ENTRY {
            return Err("invalid index sizes".into());
        }
        let obj = Self { bytes, n, segments };
        let mut first = 0;
        let mut payload = HEADER + ENTRY * segments;
        for i in 0..segments {
            let at = obj.at(i);
            let base = word(&obj.bytes, at + 8)?;
            let off = word(&obj.bytes, at + 16)?;
            let len = word(&obj.bytes, at + 24)?;
            if word(&obj.bytes, at)? != first || off != payload {
                return Err("non-contiguous segment directory".into());
            }
            let s = obj.get(i)?;
            if base.checked_add(s.first) != Some(first)
                || s.last >= batch
                || base.checked_add(s.last).filter(|&x| x < n).is_none()
            {
                return Err("invalid segment positions".into());
            }
            first = base + s.last + 1;
            payload += len;
        }
        if first != n || payload != obj.bytes.len() {
            return Err("incomplete indexed frame".into());
        }
        Ok(obj)
    }
    fn compress(values: &[Value], batch_size: usize) -> Result<Self> {
        if batch_size == 0 || batch_size > i64::MAX as usize {
            return Err("batch size must be positive and bounded".into());
        }
        let mut payload = Vec::new();
        let mut entries = Vec::<[usize; 4]>::new();
        for (batch, src) in values.chunks(batch_size).enumerate() {
            let frame = encode(src, false)?;
            let mut r = Reader {
                bytes: &frame,
                pos: 16,
            };
            let segments = r.u64()? as usize;
            for _ in 0..segments {
                let begin = r.pos;
                r.take(1)?;
                let first = r.u64()? as usize;
                r.u64()?;
                r.take(2 * std::mem::size_of::<Value>())?;
                for _ in 0..3 {
                    r.bytes()?;
                }
                let length = r.pos - begin;
                entries.push([
                    batch * batch_size + first,
                    batch * batch_size,
                    payload.len(),
                    length,
                ]);
                payload.extend_from_slice(&frame[begin..r.pos]);
            }
            if r.pos != frame.len() {
                return Err("unexpected native frame suffix".into());
            }
        }
        let base = HEADER + ENTRY * entries.len();
        let mut bytes = vec![0u8; base];
        bytes[..8].copy_from_slice(MAGIC);
        put(&mut bytes, 8, values.len());
        put(&mut bytes, 16, entries.len());
        put(&mut bytes, 24, batch_size);
        for (i, entry) in entries.iter().enumerate() {
            for (j, &value) in entry.iter().enumerate() {
                put(
                    &mut bytes,
                    HEADER + ENTRY * i + 8 * j,
                    if j == 2 { base + value } else { value },
                );
            }
        }
        bytes.extend_from_slice(&payload);
        Self::new(bytes)
    }
    fn read(&self, start: usize, dst: &mut [Value]) -> Result<()> {
        if start
            .checked_add(dst.len())
            .filter(|&end| end <= self.n)
            .is_none()
        {
            return Err("invalid query range".into());
        }
        if dst.is_empty() {
            return Ok(());
        }
        let mut lo = 0;
        let mut hi = self.segments;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if word(&self.bytes, self.at(mid))? <= start {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut id = lo.checked_sub(1).ok_or("missing segment")?;
        let mut cursor = start;
        let end = start + dst.len();
        while cursor < end {
            let at = self.at(id);
            let base = word(&self.bytes, at + 8)?;
            let s = self.get(id)?;
            let last = (base + s.last + 1).min(end);
            read_segment(
                &s,
                cursor - base - s.first,
                last - cursor,
                &mut dst[cursor - start..last - start],
            )?;
            cursor = last;
            id += 1;
        }
        Ok(())
    }
}

fn model_grid(s: &Segment<'_>, model_n: usize, lo: usize, hi: usize) -> Result<Vec<Value>> {
    if lo >= hi {
        return Ok(Vec::new());
    }
    let decoded_n = if s.model == models::GORILLA_ID {
        hi
    } else {
        hi - lo
    };
    let mut ids = UnivariateIdBuilder::with_capacity(decoded_n);
    let mut values = ValueBuilder::with_capacity(decoded_n);
    if s.model == models::GORILLA_ID {
        // Original Gorilla/Macaque decoder consumes preceding values. Bound it
        // to the requested prefix, never a preceding unrelated segment.
        let times: Vec<i64> = (s.first as i64..(s.first + hi) as i64).collect();
        models::gorilla::grid(0, s.values, &mut ids, &times, &mut values, None);
        return Ok(values.values_slice()[lo..hi].to_vec());
    }
    let times: Vec<i64> = ((s.first + lo) as i64..(s.first + hi) as i64).collect();
    match s.model {
        models::PMC_MEAN_ID => models::pmc_mean::grid(
            0,
            CompressedSegmentBuilder::decode_values_for_pmc_mean(s.min, s.max, s.values),
            &mut ids,
            &times,
            &mut values,
        ),
        models::SWING_ID => {
            let (first, last) =
                CompressedSegmentBuilder::decode_values_for_swing(s.min, s.max, s.values);
            models::swing::grid(
                0,
                s.first as i64,
                (s.first + model_n - 1) as i64,
                first,
                last,
                &mut ids,
                &times,
                &mut values,
            );
        }
        _ => return Err("unknown model".into()),
    }
    Ok(values.values_slice().to_vec())
}
fn read_segment(s: &Segment<'_>, lo: usize, count: usize, dst: &mut [Value]) -> Result<()> {
    let n = s.last - s.first + 1;
    let residual_n = s.residuals.last().copied().unwrap_or(0) as usize;
    let model_n = n - residual_n;
    let hi = lo + count;
    if hi > n || dst.len() != count {
        return Err("invalid segment query".into());
    }
    if lo < model_n {
        let values = model_grid(s, model_n, lo, hi.min(model_n))?;
        for (out, &value) in dst.iter_mut().zip(&values) {
            *out = value;
        }
    }
    if hi > model_n {
        let last = model_grid(s, model_n, model_n - 1, model_n)?[0];
        let needed = hi - model_n;
        let times: Vec<i64> = ((s.first + model_n) as i64..(s.first + hi) as i64).collect();
        let mut ids = UnivariateIdBuilder::with_capacity(needed);
        let mut values = ValueBuilder::with_capacity(needed);
        models::gorilla::grid(
            0,
            &s.residuals[..s.residuals.len() - 1],
            &mut ids,
            &times,
            &mut values,
            Some(last),
        );
        let offset = lo.saturating_sub(model_n);
        let out_start = model_n.saturating_sub(lo);
        for (out, &value) in dst[out_start..]
            .iter_mut()
            .zip(&values.values_slice()[offset..needed])
        {
            *out = value;
        }
    }
    Ok(())
}

fn slice_ok<T>(ptr: *const T, len: usize) -> bool {
    len <= isize::MAX as usize / std::mem::size_of::<T>()
        && (len == 0 || (!ptr.is_null() && (ptr as usize) % std::mem::align_of::<T>() == 0))
}
#[no_mangle]
pub unsafe extern "C" fn gef_fauna_indexed_compress(
    src: *const Value,
    n: usize,
    batch: usize,
    out: *mut *mut c_void,
) -> i32 {
    ffi_call(|| {
        if !slice_ok(src, n) || out.is_null() {
            return Err("invalid indexed compression arguments".into());
        }
        let values = if n == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(src, n)
        };
        let object = Indexed::compress(values, batch)?;
        *out = Box::into_raw(Box::new(object)).cast();
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn gef_fauna_indexed_open(
    src: *const u8,
    len: usize,
    out: *mut *mut c_void,
) -> i32 {
    ffi_call(|| {
        if !slice_ok(src, len) || out.is_null() {
            return Err("invalid indexed open arguments".into());
        }
        let bytes = if len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(src, len)
        };
        *out = Box::into_raw(Box::new(Indexed::new(bytes.to_vec())?)).cast();
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn gef_fauna_indexed_info(
    handle: *const c_void,
    n: *mut usize,
    bytes: *mut *const u8,
    len: *mut usize,
) -> i32 {
    ffi_call(|| {
        if handle.is_null() || n.is_null() || bytes.is_null() || len.is_null() {
            return Err("invalid indexed info arguments".into());
        }
        let x = &*handle.cast::<Indexed>();
        *n = x.n;
        *bytes = x.bytes.as_ptr();
        *len = x.bytes.len();
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn gef_fauna_indexed_read(
    handle: *const c_void,
    start: usize,
    n: usize,
    dst: *mut Value,
) -> i32 {
    ffi_call(|| {
        if handle.is_null() || !slice_ok(dst, n) {
            return Err("invalid indexed read arguments".into());
        }
        let dst = if n == 0 {
            &mut []
        } else {
            std::slice::from_raw_parts_mut(dst, n)
        };
        (&*handle.cast::<Indexed>()).read(start, dst)
    })
}
#[no_mangle]
pub unsafe extern "C" fn gef_fauna_indexed_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle.cast::<Indexed>()));
    }
}
