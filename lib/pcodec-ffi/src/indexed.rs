//! Indexed container over the ORIGINAL wrapped Pco chunk/page APIs.
//! No decoded values or expanded decoder tables persist between queries.
use super::{guard, input, output, valid_slice, CODEC_ERROR, INVALID, OK};
use pco::data_types::Number;
use pco::wrapped::{ChunkDecompressor, FileCompressor, FileDecompressor};
use pco::{ChunkConfig, PagingSpec};
use std::ffi::c_void;

const HEADER: usize = 64;
const CHUNK: usize = 48;
const PAGE: usize = 40;
const MAGIC: &[u8; 8] = b"GFPCPG01";
type Result<T> = std::result::Result<T, String>;
fn put(bytes: &mut [u8], at: usize, value: usize) {
    bytes[at..at + 8].copy_from_slice(&(value as u64).to_le_bytes());
}
fn word(bytes: &[u8], at: usize) -> Result<usize> {
    let b = bytes
        .get(at..at.checked_add(8).ok_or("offset overflow")?)
        .ok_or("truncated index")?;
    usize::try_from(u64::from_le_bytes(b.try_into().unwrap())).map_err(|_| "index overflow".into())
}
fn part(bytes: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    bytes
        .get(at..at.checked_add(len).ok_or("offset overflow")?)
        .ok_or_else(|| "invalid payload bounds".into())
}
pub struct Indexed {
    bytes: Vec<u8>,
    n: usize,
    chunks: usize,
    pages: usize,
    kind: usize,
}
impl Indexed {
    fn page_at(&self, id: usize) -> usize {
        HEADER + CHUNK * self.chunks + PAGE * id
    }
    fn chunk_at(&self, id: usize) -> usize {
        HEADER + CHUNK * id
    }
    fn new(bytes: Vec<u8>, kind: usize) -> Result<Self> {
        if bytes.len() < HEADER || &bytes[..8] != MAGIC {
            return Err("wrong indexed Pco format".into());
        }
        let (n, chunks, pages) = (word(&bytes, 8)?, word(&bytes, 24)?, word(&bytes, 32)?);
        if word(&bytes, 16)? != kind || word(&bytes, 56)? != bytes.len() || ![1, 2].contains(&kind)
        {
            return Err("wrong type or frame length".into());
        }
        if chunks > bytes.len() / CHUNK || pages > bytes.len() / PAGE {
            return Err("invalid directory sizes".into());
        }
        let directory_end = HEADER
            .checked_add(CHUNK * chunks)
            .and_then(|x| x.checked_add(PAGE * pages))
            .ok_or("index overflow")?;
        if word(&bytes, 40)? != directory_end {
            return Err("invalid header offset".into());
        }
        part(&bytes, directory_end, word(&bytes, 48)?)?;
        let obj = Self {
            bytes,
            n,
            chunks,
            pages,
            kind,
        };
        let mut expected_n = 0;
        let mut expected_page = 0;
        let mut payload = directory_end + word(&obj.bytes, 48)?;
        for c in 0..chunks {
            let at = obj.chunk_at(c);
            let first = word(&obj.bytes, at)?;
            let count = word(&obj.bytes, at + 8)?;
            let offset = word(&obj.bytes, at + 16)?;
            let length = word(&obj.bytes, at + 24)?;
            let first_page = word(&obj.bytes, at + 32)?;
            let n_pages = word(&obj.bytes, at + 40)?;
            if first != expected_n
                || count == 0
                || offset != payload
                || first_page != expected_page
                || n_pages == 0
                || n_pages > pages - expected_page
            {
                return Err("invalid chunk directory".into());
            }
            part(&obj.bytes, offset, length)?;
            payload += length;
            let mut chunk_n = 0usize;
            for id in first_page..first_page + n_pages {
                let p = obj.page_at(id);
                let start = word(&obj.bytes, p)?;
                let len = word(&obj.bytes, p + 8)?;
                let owner = word(&obj.bytes, p + 16)?;
                let off = word(&obj.bytes, p + 24)?;
                let size = word(&obj.bytes, p + 32)?;
                if start != expected_n
                    || len == 0
                    || len > n.saturating_sub(expected_n)
                    || owner != c
                    || off != payload
                {
                    return Err("invalid page directory".into());
                }
                part(&obj.bytes, off, size)?;
                payload += size;
                chunk_n += len;
                expected_n += len;
            }
            if chunk_n != count {
                return Err("chunk count mismatch".into());
            }
            expected_page += n_pages;
        }
        if expected_n != n || expected_page != pages || payload != obj.bytes.len() {
            return Err("incomplete directory".into());
        }
        Ok(obj)
    }
    fn compress<T: Number>(
        values: &[T],
        chunk_size: usize,
        page_size: usize,
        kind: usize,
    ) -> Result<Self> {
        if chunk_size == 0 || page_size == 0 {
            return Err("chunk/page size must be positive".into());
        }
        // Same balanced chunk/page partitioning as the authors' pcopage benchmark.
        let chunks = PagingSpec::EqualPagesUpTo(chunk_size)
            .n_per_page(values.len())
            .map_err(|e| e.to_string())?;
        let config = ChunkConfig::default().with_paging_spec(PagingSpec::EqualPagesUpTo(page_size));
        let mut pages = 0usize;
        for &n in &chunks {
            pages += config
                .paging_spec
                .n_per_page(n)
                .map_err(|e| e.to_string())?
                .len();
        }
        let header_at = HEADER + CHUNK * chunks.len() + PAGE * pages;
        let mut bytes = vec![0u8; header_at];
        bytes[..8].copy_from_slice(MAGIC);
        put(&mut bytes, 8, values.len());
        put(&mut bytes, 16, kind);
        put(&mut bytes, 24, chunks.len());
        put(&mut bytes, 32, pages);
        put(&mut bytes, 40, header_at);
        let fc = FileCompressor::default();
        fc.write_header(&mut bytes).map_err(|e| e.to_string())?;
        let header_len = bytes.len() - header_at;
        put(&mut bytes, 48, header_len);
        let mut first = 0;
        let mut page_id = 0;
        for (c, &n) in chunks.iter().enumerate() {
            let mut cc = fc
                .chunk_compressor(&values[first..first + n], &config)
                .map_err(|e| e.to_string())?;
            let page_ns = cc.n_per_page();
            let meta_at = bytes.len();
            cc.write_meta(&mut bytes).map_err(|e| e.to_string())?;
            let meta_len = bytes.len() - meta_at;
            let at = HEADER + CHUNK * c;
            for (field, value) in [first, n, meta_at, meta_len, page_id, page_ns.len()]
                .into_iter()
                .enumerate()
            {
                put(&mut bytes, at + 8 * field, value);
            }
            for (i, &page_n) in page_ns.iter().enumerate() {
                let off = bytes.len();
                cc.write_page(i, &mut bytes).map_err(|e| e.to_string())?;
                let size = bytes.len() - off;
                let p = HEADER + CHUNK * chunks.len() + PAGE * page_id;
                for (field, value) in [first, page_n, c, off, size].into_iter().enumerate() {
                    put(&mut bytes, p + 8 * field, value);
                }
                first += page_n;
                page_id += 1;
            }
        }
        let len = bytes.len();
        put(&mut bytes, 56, len);
        Self::new(bytes, kind)
    }
    fn read<T: Number + Default>(&self, start: usize, dst: &mut [T], kind: usize) -> Result<()> {
        if kind != self.kind
            || start
                .checked_add(dst.len())
                .filter(|&x| x <= self.n)
                .is_none()
        {
            return Err("invalid query".into());
        }
        if dst.is_empty() {
            return Ok(());
        }
        let (fd, rest) = FileDecompressor::new(part(
            &self.bytes,
            word(&self.bytes, 40)?,
            word(&self.bytes, 48)?,
        )?)
        .map_err(|e| e.to_string())?;
        if !rest.is_empty() {
            return Err("invalid Pco header".into());
        }
        let mut lo = 0;
        let mut hi = self.pages;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if word(&self.bytes, self.page_at(mid))? <= start {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut page = lo.checked_sub(1).ok_or("missing page")?;
        let end = start + dst.len();
        let mut cursor = start;
        // Reuse native chunk metadata across pages WITHIN this range query.
        // No uncharged persistent expanded tables or decoded-value cache.
        let mut current: Option<(usize, ChunkDecompressor<T>)> = None;
        let mut scratch = vec![T::default(); 256];
        while cursor < end {
            let p = self.page_at(page);
            let first = word(&self.bytes, p)?;
            let n = word(&self.bytes, p + 8)?;
            let c = word(&self.bytes, p + 16)?;
            if current.as_ref().map(|x| x.0) != Some(c) {
                let at = self.chunk_at(c);
                let meta = part(
                    &self.bytes,
                    word(&self.bytes, at + 16)?,
                    word(&self.bytes, at + 24)?,
                )?;
                let (cd, rest) = fd
                    .chunk_decompressor::<T, _>(meta)
                    .map_err(|e| e.to_string())?;
                if !rest.is_empty() {
                    return Err("trailing chunk metadata".into());
                }
                current = Some((c, cd));
            }
            let bytes = part(
                &self.bytes,
                word(&self.bytes, p + 24)?,
                word(&self.bytes, p + 32)?,
            )?;
            let mut pd = current
                .as_mut()
                .unwrap()
                .1
                .page_decompressor(bytes, n)
                .map_err(|e| e.to_string())?;
            let wanted_end = end.min(first + n);
            let mut decoded = 0;
            while first + decoded < wanted_end {
                // Feed a contiguous requested span directly to the original
                // decoder, as upstream's full-page benchmark does. Scratch is
                // only needed for skipped prefixes and partial end batches.
                let pos = first + decoded;
                if pos >= cursor {
                    let available = wanted_end - pos;
                    let direct = if wanted_end == first + n {
                        available
                    } else {
                        available / 256 * 256
                    };
                    if direct > 0 {
                        let progress = pd
                            .read(&mut dst[pos - start..pos - start + direct])
                            .map_err(|e| e.to_string())?;
                        if progress.n_processed != direct {
                            return Err("incomplete native span".into());
                        }
                        decoded += direct;
                        continue;
                    }
                }
                // Upstream requires full 256-value batches, except at page end.
                let take = (n - decoded).min(256);
                let progress = pd.read(&mut scratch[..take]).map_err(|e| e.to_string())?;
                if progress.n_processed != take {
                    return Err("incomplete native batch".into());
                }
                let a = cursor.max(first + decoded);
                let b = wanted_end.min(first + decoded + take);
                if b > a {
                    dst[a - start..b - start]
                        .copy_from_slice(&scratch[a - first - decoded..b - first - decoded]);
                }
                decoded += take;
            }
            if decoded == n && !pd.into_src().is_empty() {
                return Err("trailing page data".into());
            }
            cursor = wanted_end;
            page += 1;
        }
        Ok(())
    }
}

#[no_mangle]
pub unsafe extern "C" fn gef_pco_indexed_open(
    src: *const u8,
    len: usize,
    kind: usize,
    out: *mut *mut c_void,
) -> i32 {
    guard(|| {
        if !valid_slice(src, len) || out.is_null() {
            return INVALID;
        }
        match Indexed::new(input(src, len).to_vec(), kind) {
            Ok(x) => {
                *out = Box::into_raw(Box::new(x)).cast();
                OK
            }
            Err(_) => CODEC_ERROR,
        }
    })
}
#[no_mangle]
pub unsafe extern "C" fn gef_pco_indexed_info(
    handle: *const c_void,
    n: *mut usize,
    bytes: *mut *const u8,
    len: *mut usize,
) -> i32 {
    if handle.is_null() || n.is_null() || bytes.is_null() || len.is_null() {
        return INVALID;
    }
    let x = &*handle.cast::<Indexed>();
    *n = x.n;
    *bytes = x.bytes.as_ptr();
    *len = x.bytes.len();
    OK
}
#[no_mangle]
pub unsafe extern "C" fn gef_pco_indexed_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle.cast::<Indexed>()));
    }
}
macro_rules! export_indexed {
    ($ty:ty,$kind:expr,$encode:ident,$read:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $encode(
            src: *const $ty,
            n: usize,
            chunk: usize,
            page: usize,
            out: *mut *mut c_void,
        ) -> i32 {
            guard(|| {
                if !valid_slice(src, n) || out.is_null() {
                    return INVALID;
                }
                match Indexed::compress(input(src, n), chunk, page, $kind) {
                    Ok(x) => {
                        *out = Box::into_raw(Box::new(x)).cast();
                        OK
                    }
                    Err(_) => CODEC_ERROR,
                }
            })
        }
        #[no_mangle]
        pub unsafe extern "C" fn $read(
            handle: *const c_void,
            start: usize,
            n: usize,
            dst: *mut $ty,
        ) -> i32 {
            guard(|| {
                if handle.is_null() || !valid_slice(dst, n) {
                    return INVALID;
                }
                match (&*handle.cast::<Indexed>()).read(start, output(dst, n), $kind) {
                    Ok(()) => OK,
                    Err(_) => CODEC_ERROR,
                }
            })
        }
    };
}
export_indexed!(
    f64,
    1,
    gef_pco_indexed_compress_f64,
    gef_pco_indexed_read_f64
);
export_indexed!(
    i64,
    2,
    gef_pco_indexed_compress_i64,
    gef_pco_indexed_read_i64
);
