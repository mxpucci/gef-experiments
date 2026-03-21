use mem_dbg::MemSize;
use sux::list::comp_int_list::CompIntList;
use sux::list::prefix_sum_int_list::PrefixSumIntList;
use std::slice;
use value_traits::slices::SliceByValue;

// ============================================================================
// CompIntList FFI
// ============================================================================

pub struct CompIntListHandle {
    inner: CompIntList<u64>,
}

#[no_mangle]
pub extern "C" fn comp_int_list_new(
    values_ptr: *const u64,
    len: usize,
    min_val: u64,
) -> *mut CompIntListHandle {
    if values_ptr.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    let values = unsafe { slice::from_raw_parts(values_ptr, len) };
    let list = CompIntList::new(min_val, &values.to_vec());
    Box::into_raw(Box::new(CompIntListHandle { inner: list }))
}

#[no_mangle]
pub extern "C" fn comp_int_list_get(handle: *const CompIntListHandle, index: usize) -> u64 {
    if handle.is_null() {
        return 0;
    }
    let handle = unsafe { &*handle };
    handle.inner.index_value(index)
}

#[no_mangle]
pub extern "C" fn comp_int_list_len(handle: *const CompIntListHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let handle = unsafe { &*handle };
    handle.inner.len()
}

#[no_mangle]
pub extern "C" fn comp_int_list_mem_size(handle: *const CompIntListHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let handle = unsafe { &*handle };
    handle.inner.mem_size(mem_dbg::SizeFlags::default())
}

#[no_mangle]
pub extern "C" fn comp_int_list_get_range(
    handle: *const CompIntListHandle,
    start: usize,
    count: usize,
    output_ptr: *mut u64,
) {
    if handle.is_null() || output_ptr.is_null() {
        return;
    }
    let handle = unsafe { &*handle };
    let output = unsafe { slice::from_raw_parts_mut(output_ptr, count) };
    for i in 0..count {
        output[i] = handle.inner.index_value(start + i);
    }
}

#[no_mangle]
pub extern "C" fn comp_int_list_free(handle: *mut CompIntListHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

// ============================================================================
// PrefixSumIntList FFI
// ============================================================================

pub struct PrefixSumIntListHandle {
    inner: PrefixSumIntList<u64>,
}

#[no_mangle]
pub extern "C" fn prefix_sum_int_list_new(
    values_ptr: *const u64,
    len: usize,
) -> *mut PrefixSumIntListHandle {
    if values_ptr.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    let values = unsafe { slice::from_raw_parts(values_ptr, len) };
    let list = PrefixSumIntList::new(&values.to_vec());
    Box::into_raw(Box::new(PrefixSumIntListHandle { inner: list }))
}

#[no_mangle]
pub extern "C" fn prefix_sum_int_list_get(
    handle: *const PrefixSumIntListHandle,
    index: usize,
) -> u64 {
    if handle.is_null() {
        return 0;
    }
    let handle = unsafe { &*handle };
    handle.inner.index_value(index)
}

#[no_mangle]
pub extern "C" fn prefix_sum_int_list_len(handle: *const PrefixSumIntListHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let handle = unsafe { &*handle };
    handle.inner.len()
}

#[no_mangle]
pub extern "C" fn prefix_sum_int_list_mem_size(handle: *const PrefixSumIntListHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let handle = unsafe { &*handle };
    handle.inner.mem_size(mem_dbg::SizeFlags::default())
}

#[no_mangle]
pub extern "C" fn prefix_sum_int_list_get_range(
    handle: *const PrefixSumIntListHandle,
    start: usize,
    count: usize,
    output_ptr: *mut u64,
) {
    if handle.is_null() || output_ptr.is_null() {
        return;
    }
    let handle = unsafe { &*handle };
    let output = unsafe { slice::from_raw_parts_mut(output_ptr, count) };
    for i in 0..count {
        output[i] = handle.inner.index_value(start + i);
    }
}

#[no_mangle]
pub extern "C" fn prefix_sum_int_list_free(handle: *mut PrefixSumIntListHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}
