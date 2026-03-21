#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

// CompIntList: compact list of integers >= min, with random access via EF delimiters
typedef struct CompIntListHandle CompIntListHandle;

CompIntListHandle* comp_int_list_new(const uint64_t* values, size_t len, uint64_t min_val);
uint64_t comp_int_list_get(const CompIntListHandle* handle, size_t index);
size_t comp_int_list_len(const CompIntListHandle* handle);
size_t comp_int_list_mem_size(const CompIntListHandle* handle);
void comp_int_list_get_range(const CompIntListHandle* handle, size_t start, size_t count, uint64_t* output);
void comp_int_list_free(CompIntListHandle* handle);

// PrefixSumIntList: compact list of non-negative integers stored as prefix sums in EF
typedef struct PrefixSumIntListHandle PrefixSumIntListHandle;

PrefixSumIntListHandle* prefix_sum_int_list_new(const uint64_t* values, size_t len);
uint64_t prefix_sum_int_list_get(const PrefixSumIntListHandle* handle, size_t index);
size_t prefix_sum_int_list_len(const PrefixSumIntListHandle* handle);
size_t prefix_sum_int_list_mem_size(const PrefixSumIntListHandle* handle);
void prefix_sum_int_list_get_range(const PrefixSumIntListHandle* handle, size_t start, size_t count, uint64_t* output);
void prefix_sum_int_list_free(PrefixSumIntListHandle* handle);

} // extern "C"
