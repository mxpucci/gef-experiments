#include "benchmark_common.hpp"
#include "Falcon/CompressorFalcon.hpp"
#include "Falcon/DecompressorFalcon.hpp"
#include <bit>
#include <chrono>
#include <iostream>
#ifdef __FAST_MATH__
#error Falcon must be compiled with strict IEEE arithmetic
#endif

BenchmarkResult benchmark_falcon(const BenchmarkData &bench_data,
                                 const std::vector<size_t> &range_sizes,
                                 size_t block_size) { // Default to 1000 to match your proposal
    using T = double;
    BenchmarkResult result;
    result.compressor = "Falcon";
    result.dataset = bench_data.filename;
    
    // Use double data
    const auto& data = bench_data.double_data;
    
    result.num_values = data.size();
    result.uncompressed_bits = bench_data.uncompressed_bits;
    
    const size_t n = data.size();
    const size_t num_blocks = (n + block_size - 1) / block_size;
    
    // -------------------------------------------------------------------------
    // 2. Compression (Independent Blocks)
    // -------------------------------------------------------------------------
    size_t total_compressed_bits = 0;
    std::vector<std::vector<uint8_t>> compressed_blocks(num_blocks);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        // Identify range for this block
        const size_t start_idx = ib * block_size;
        const size_t end_idx = std::min(start_idx + block_size, n);
        const size_t current_bs = end_idx - start_idx;
        
        // Initialize Compressor for JUST this block
        // Note: We pass current_bs so Falcon writes the correct header count
        // We also pass decimals to help Falcon avoid raw fallback on tricky values
        CompressorFalcon<T> cmpr(data[start_idx], current_bs, static_cast<int>(bench_data.decimals));
        
        for (size_t k = 1; k < current_bs; ++k) {
            cmpr.addValue(data[start_idx + k]);
        }
        cmpr.close();
        
        // Store results
        total_compressed_bits += cmpr.getSize();
        compressed_blocks[ib] = cmpr.getOut(); 
    }
    compiler_barrier();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    
    result.compressed_bits = total_compressed_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (compression_time_ns / 1e9);
    
    // -------------------------------------------------------------------------
    // 3. Full Decompression (Sanity Check & Throughput)
    // -------------------------------------------------------------------------
    std::vector<T> decompressed(n);
    volatile T decomp_checksum = 0;
    t1 = std::chrono::high_resolution_clock::now();
    
    size_t output_offset = 0;
    for (size_t ib = 0; ib < num_blocks; ++ib) {
        size_t expected_count = std::min(block_size, n - ib * block_size);
        
        // Decompress independent block
        DecompressorFalcon<T> dcmpr(compressed_blocks[ib], expected_count);
        
        // Falcon Decompressor initializes storedValue with the first item immediately
        decompressed[output_offset++] = dcmpr.storedValue;
        
        for (size_t k = 1; k < expected_count; ++k) {
            if (dcmpr.hasNext()) {
                decompressed[output_offset++] = dcmpr.storedValue;
            } else {
                break;
            }
        }
        decomp_checksum += decompressed[ib * block_size];
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    do_not_optimize(decomp_checksum);
    auto decompression_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.decompression_throughput_mbs = (n * sizeof(T) / 1024.0 / 1024.0) / (decompression_time_ns / 1e9);
    

    for (size_t i = 0; i < n; ++i)
        result.roundtrip_mismatches += std::bit_cast<uint64_t>(data[i]) != std::bit_cast<uint64_t>(decompressed[i]);
    result.validation_status = result.roundtrip_mismatches ? "bitwise_failed" : "bitwise_passed";
    result.access_layout = "independent_blocks";
    result.compression_unit_values = block_size;
    if (result.roundtrip_mismatches)
        std::cerr << "WARNING: Falcon: " << result.roundtrip_mismatches << " bitwise round-trip mismatches\n";

    // -------------------------------------------------------------------------
    // 4. Random Access (Decompress ONLY the target block)
    // -------------------------------------------------------------------------
    const size_t num_ra_queries = 10000;
    
    t1 = std::chrono::high_resolution_clock::now();
    volatile T ra_sum = 0;
    size_t q_count = 0;
    
    for (auto idx : bench_data.random_indices) {
        if (q_count++ >= num_ra_queries) break;

        size_t ib = idx / block_size;
        size_t offset_in_block = idx % block_size;
        size_t expected_count = std::min(block_size, n - ib * block_size);
        
        // Instantiate decompressor only for the required block
        DecompressorFalcon<T> dcmpr(compressed_blocks[ib], expected_count);
        
        // Scan to the specific offset
        size_t current_pos = 0;
        if (offset_in_block == 0) {
            ra_sum += dcmpr.storedValue;
        } else {
            // hasNext() advances to the next value and updates storedValue
            while (current_pos < offset_in_block && dcmpr.hasNext()) {
                current_pos++;
            }
            ra_sum += dcmpr.storedValue;
        }
    }
    compiler_barrier();
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(ra_sum);
    
    auto ra_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.random_access_ns = static_cast<double>(ra_time_ns) / num_ra_queries;
    result.random_access_mbs = (num_ra_queries * sizeof(T) / 1024.0 / 1024.0) / (ra_time_ns / 1e9);

    // -------------------------------------------------------------------------
    // 5. Range Queries
    // -------------------------------------------------------------------------
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) continue;
        
        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<T> out_buffer(range);
        volatile T range_checksum = 0;
        
        t1 = std::chrono::high_resolution_clock::now();
        size_t q_count_range = 0;
        for (auto start_idx : range_indices) {
            if (q_count_range++ >= num_range_queries) break;
            
            size_t start_block = start_idx / block_size;
            size_t end_block = (start_idx + range - 1) / block_size;
            
            size_t out_pos = 0;
            
            // We only decompress the blocks overlapping the range
            for (size_t ib = start_block; ib <= end_block; ++ib) {
                size_t expected_count = std::min(block_size, n - ib * block_size);
                DecompressorFalcon<T> dcmpr(compressed_blocks[ib], expected_count);
                
                size_t block_start_global = ib * block_size;
                
                // Determine how many items to skip in this block (only for the first block)
                size_t skip = (ib == start_block) ? (start_idx - block_start_global) : 0;
                
                size_t current_in_block = 0;
                
                // Fast skip
                // Note: storedValue is the 0-th element. 
                // We need to call hasNext 'skip' times to put 'storedValue' at index 'skip'
                while(current_in_block < skip && dcmpr.hasNext()) {
                    current_in_block++;
                }
                
                // Copy
                while (out_pos < range) {
                    out_buffer[out_pos++] = dcmpr.storedValue;
                    current_in_block++;
                    
                    // Boundary check: if we hit end of block or end of range, break
                    if (current_in_block >= expected_count) break;
                    if (!dcmpr.hasNext()) break; 
                }
            }
            range_checksum += out_buffer[0];
            do_not_optimize(out_buffer);
        }
        compiler_barrier();
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(range_checksum);
        
        auto range_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = ((range * sizeof(T)) * num_range_queries / 1024.0 / 1024.0) / (range_time_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
    }
    
    return result;
}

