// ALP benchmark - uses the ALP library for floating-point compression.

#include "benchmark_common.hpp"
#include "alp.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

#ifdef __FAST_MATH__
#error "ALP adapter requires strict floating-point semantics"
#endif

namespace {

// Partial last vector, following the authors' DuckDB integration: encode it
// with its real value count and bit-pack only the needed prefix. FastLanes
// puts value i in lane i % L (L = 1024/W lanes of W-bit words), so values
// [0,n) live in the first ceil(ceil(n/L)*bw/W) words of every lane.
inline size_t packed_words(size_t n, unsigned bw, unsigned w) {
    const size_t lanes = alp::config::VECTOR_SIZE / w, rows = (n + lanes - 1) / lanes;
    return lanes * ((rows * bw + w - 1) / w) * w / 64;
}

// Metadata structure mirroring the authors' alp_m struct (but adapted for in-memory)
// See: lib/ALP/publication/source_code/bench_end_to_end/src/common/runtime/Import.cpp
struct VectorMetadata {
    bool rd = false;
    size_t rd_group = 0;
    uint8_t bit_width;
    uint8_t factor;
    uint8_t exponent;
    int64_t ffor_base;
    uint16_t exc_count;
    uint16_t count = alp::config::VECTOR_SIZE; // values; derived from n, not stored

    // Offsets into the respective data vectors
    size_t ffor_offset;
    size_t exc_val_offset;
    size_t exc_pos_offset;
};

// Helper to decode a single vector using separate data streams
// Uses the fused FALP function as per official ALP end-to-end benchmark
// See: lib/ALP/publication/source_code/bench_end_to_end/src/benchmarks/alp/queries/q1.cpp
inline void decode_vector(
    const VectorMetadata& meta,
    const uint64_t* ffor_stream,
    const double* exc_val_stream,
    const uint16_t* exc_pos_stream,
    const uint16_t* rd_exc_stream,
    double* out_vec,
    std::vector<alp::state<double>>& rowgroups)
{
    // The partial vector stores a packed prefix; zero-fill the rest of a
    // full FastLanes block so the original unpackers can be used unchanged.
    alignas(64) uint64_t padded[16 * (64 + 16)];
    const bool partial = meta.count < alp::config::VECTOR_SIZE;
    if (meta.rd) {
        auto& stt = rowgroups[meta.rd_group];
        alignas(64) uint64_t right[alp::config::VECTOR_SIZE];
        alignas(64) uint16_t left[alp::config::VECTOR_SIZE];
        const auto* packed = ffor_stream + meta.ffor_offset;
        if (partial) {
            const size_t rw = packed_words(meta.count, stt.right_bit_width, 64);
            std::fill(std::begin(padded), std::end(padded), 0);
            std::copy(packed, packed + rw, padded);
            std::copy(packed + rw, packed + rw + packed_words(meta.count, stt.left_bit_width, 16),
                      padded + 16 * stt.right_bit_width);
            packed = padded;
        }
        unffor::unffor(packed, right, stt.right_bit_width, &stt.right_for_base);
        unffor::unffor(reinterpret_cast<const uint16_t*>(packed + 16 * stt.right_bit_width),
                       left, stt.left_bit_width, &stt.left_for_base);
        auto count=meta.exc_count;
        alp::rd_encoder<double>::decode(out_vec,right,left,
            const_cast<uint16_t*>((meta.exc_count ? rd_exc_stream+meta.exc_val_offset : nullptr)),
            const_cast<uint16_t*>((meta.exc_count ? exc_pos_stream+meta.exc_pos_offset : nullptr)),&count,stt);
        return;
    }
    // 1. FALP: Fused bit-unpacking + ALP decode in one pass (as per official implementation)
    static const uint64_t zero=0;
    const uint64_t* packed_ptr = meta.bit_width ? ffor_stream + meta.ffor_offset : &zero;
    if (partial && meta.bit_width) {
        std::fill(std::begin(padded), std::end(padded), 0);
        std::copy(packed_ptr, packed_ptr + packed_words(meta.count, meta.bit_width, 64), padded);
        packed_ptr = padded;
    }
    uint64_t base_u64 = static_cast<uint64_t>(meta.ffor_base);

    generated::falp::fallback::scalar::falp(
        packed_ptr,
        out_vec,
        meta.bit_width,
        &base_u64,
        meta.factor,
        meta.exponent
    );

    // 2. Patch Exceptions (as per official implementation)
    if (meta.exc_count > 0) {
        const double* exceptions = exc_val_stream + meta.exc_val_offset;
        const uint16_t* positions = exc_pos_stream + meta.exc_pos_offset;

        alp::exp_c_t ec = meta.exc_count;
        alp::decoder<double>::patch_exceptions(out_vec, exceptions, positions, &ec);
    }
}

} // anonymous namespace

BenchmarkResult benchmark_alp(const BenchmarkData &bench_data,
                              const std::vector<size_t> &range_sizes) {
    BenchmarkResult result;
    result.compressor = "ALP";
    result.dataset = bench_data.filename;

    const auto& data = bench_data.double_data;
    const size_t n = data.size();
    if (!n) throw std::invalid_argument("ALP requires a nonempty column");

    result.num_values = n;
    result.uncompressed_bits = bench_data.uncompressed_bits;

    constexpr size_t VEC = alp::config::VECTOR_SIZE;
    const size_t num_vecs = (n + VEC - 1) / VEC; // last one may be partial

    // Streams follow the authors' end-to-end integration; RD exceptions
    // use their native uint16 representation in a separate stream.
    // 1. Metadata Stream
    std::vector<VectorMetadata> metadata_stream;
    metadata_stream.reserve(num_vecs);

    // 2. FFOR Bitstream (Packed Integers)
    std::vector<uint64_t> ffor_stream;
    ffor_stream.reserve(n); // Conservative estimate

    // 3. Exception Values Stream
    std::vector<double> exc_val_stream;
    exc_val_stream.reserve(n / 10);

    // 4. Exception Positions Stream
    std::vector<uint16_t> rd_exc_stream;
    std::vector<uint16_t> exc_pos_stream;
    exc_pos_stream.reserve(n / 10);

    // Sample buffer
    constexpr size_t SAMPLE_BUF_SIZE = alp::config::ROWGROUP_SIZE;
    std::vector<double> sample_buf(SAMPLE_BUF_SIZE);

    std::vector<alp::state<double>> rowgroups;
    size_t rd_vectors=0, alp_vectors=0;
    // Compression
    auto t1 = std::chrono::high_resolution_clock::now();
    {
        constexpr size_t ROWGROUP_SIZE = alp::config::ROWGROUP_SIZE;
        const size_t num_rowgroups = (n + ROWGROUP_SIZE - 1) / ROWGROUP_SIZE;

        for (size_t rg = 0; rg < num_rowgroups; ++rg) {
            const size_t rg_start = rg * ROWGROUP_SIZE;
            const size_t rg_end = std::min(rg_start + ROWGROUP_SIZE, n);

            alp::state<double> stt;
            alp::encoder<double>::init(data.data(), rg_start, n, sample_buf.data(), stt);
            if (stt.scheme == alp::Scheme::ALP_RD) {
                alp::rd_encoder<double>::init(data.data(), rg_start, n, sample_buf.data(), stt);
            }

            alp::state<double> decoder_state;
            decoder_state.actual_dictionary_size=stt.actual_dictionary_size;
            decoder_state.right_bit_width=stt.right_bit_width;
            decoder_state.left_bit_width=stt.left_bit_width;
            std::copy(std::begin(stt.left_parts_dict),std::end(stt.left_parts_dict),decoder_state.left_parts_dict);
            rowgroups.push_back(std::move(decoder_state));
            const size_t rg_vec_start = rg_start / VEC;
            const size_t rg_vec_end = (rg_end + VEC - 1) / VEC;

            for (size_t v = rg_vec_start; v < rg_vec_end; ++v) {
                const double* vec_in = data.data() + v * VEC;
                // Partial vector: the upstream kernels read 1024 values, so
                // repeat the last real value (no new min/max or exception
                // positions < count), search exp/fac over count values via
                // the upstream vector_size field, and drop pad exceptions.
                const size_t count = std::min(VEC, n - v * VEC);
                alignas(64) double padded_in[VEC];
                if (count < VEC) {
                    std::copy(vec_in, vec_in + count, padded_in);
                    std::fill(padded_in + count, padded_in + VEC, vec_in[count - 1]);
                    vec_in = padded_in;
                    stt.vector_size = static_cast<uint16_t>(count);
                }
                auto drop_padding = [&](uint16_t& c, const uint16_t* pos) {
                    while (c && pos[c - 1] >= count) --c;
                };

                alignas(64) int64_t temp_encoded[VEC];
                alignas(64) double exc_buf[VEC] = {};
                alignas(64) uint16_t pos_buf[VEC] = {};
                uint16_t exc_cnt = 0;

                alp::bw_t bw = 0;
                int64_t for_base = 0;
                uint8_t fac = 0;
                uint8_t exp = 0;

                if (stt.scheme == alp::Scheme::ALP_RD) {
                    alignas(64) uint64_t right[VEC];
                    alignas(64) uint16_t left[VEC], exceptions[VEC];
                    alp::rd_encoder<double>::encode(vec_in, exceptions, pos_buf, &exc_cnt, right, left, stt);
                    drop_padding(exc_cnt, pos_buf);
                    VectorMetadata meta{};
                    meta.rd=true; meta.rd_group=rg; meta.count=static_cast<uint16_t>(count);
                    meta.exc_count=exc_cnt;
                    meta.ffor_offset=ffor_stream.size();
                    meta.exc_val_offset=rd_exc_stream.size();
                    meta.exc_pos_offset=exc_pos_stream.size();
                    ffor_stream.resize(ffor_stream.size()+16*(stt.right_bit_width+stt.left_bit_width));
                    auto* packed=ffor_stream.data()+meta.ffor_offset;
                    ffor::ffor(right,packed,stt.right_bit_width,&stt.right_for_base);
                    ffor::ffor(left,reinterpret_cast<uint16_t*>(packed+16*stt.right_bit_width),stt.left_bit_width,&stt.left_for_base);
                    if (count < VEC) { // keep only the packed prefixes, left after right
                        const size_t rw=packed_words(count,stt.right_bit_width,64);
                        const size_t lw=packed_words(count,stt.left_bit_width,16);
                        std::memmove(packed+rw,packed+16*stt.right_bit_width,lw*sizeof(uint64_t));
                        ffor_stream.resize(meta.ffor_offset+rw+lw);
                    }
                    rd_exc_stream.insert(rd_exc_stream.end(),exceptions,exceptions+exc_cnt);
                    exc_pos_stream.insert(exc_pos_stream.end(),pos_buf,pos_buf+exc_cnt);
                    metadata_stream.push_back(std::move(meta));
                    ++rd_vectors;
                    continue;
                } else {
                    alp::encoder<double>::encode(vec_in, exc_buf, pos_buf, &exc_cnt, temp_encoded, stt);
                    drop_padding(exc_cnt, pos_buf);
                    alp::encoder<double>::analyze_ffor(temp_encoded, bw, &for_base);
                    fac = stt.fac;
                    exp = stt.exp;
                }

                // Store Metadata
                ++alp_vectors;
                VectorMetadata meta{};
                meta.bit_width = bw;
                meta.factor = fac;
                meta.exponent = exp;
                meta.ffor_base = for_base;
                meta.exc_count = exc_cnt;
                meta.count = static_cast<uint16_t>(count);
                meta.ffor_offset = ffor_stream.size(); // Index in 64-bit words
                meta.exc_val_offset = exc_val_stream.size();
                meta.exc_pos_offset = exc_pos_stream.size();
                metadata_stream.push_back(meta);

                // Store FFOR Data
                if (bw > 0) {
                    size_t ffor_words = (VEC * bw + 63) / 64;
                    size_t current_size = ffor_stream.size();
                    ffor_stream.resize(current_size + ffor_words);
                    ffor::ffor(temp_encoded, reinterpret_cast<int64_t*>(&ffor_stream[current_size]), bw, &for_base);
                    ffor_stream.resize(current_size + packed_words(count, bw, 64));
                }

                // Store Exceptions
                if (exc_cnt > 0) {
                    exc_val_stream.insert(exc_val_stream.end(), exc_buf, exc_buf + exc_cnt);
                    exc_pos_stream.insert(exc_pos_stream.end(), pos_buf, pos_buf + exc_cnt);
                }
            }
        }
    }
    // Materialize compact metadata using the upstream byte storer. Offset
    // fields mirror the authors' end-to-end three-stream integration.
    std::vector<uint8_t> stored_metadata(8 + rowgroups.size()*20 + num_vecs*37);
    alp::storer::MemStorer<> store(stored_metadata.data());
    uint64_t stored_count=n;
    store.store(&stored_count,8);
    for(size_t rg=0;rg<rowgroups.size();++rg) {
        uint8_t rd=metadata_stream[rg*alp::config::N_VECTORS_PER_ROWGROUP].rd;
        store.store(&rd,1);
        if(rd) {
            auto& st=rowgroups[rg];
            store.store(&st.right_bit_width,1);store.store(&st.left_bit_width,1);
            store.store(&st.actual_dictionary_size,1);
            store.store(st.left_parts_dict,st.actual_dictionary_size*2);
        }
    }
    for(auto& m:metadata_stream) {
        uint64_t offsets[]={m.ffor_offset,m.exc_val_offset,m.exc_pos_offset};
        store.store(offsets,sizeof(offsets));
        store.store(&m.exc_count,2);
        if(!m.rd) {
            store.store(&m.bit_width,1);store.store(&m.factor,1);store.store(&m.exponent,1);
            store.store(&m.ffor_base,8);
        }
    }
    stored_metadata.resize(store.get_size());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto comp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    // Count a packed representation, not C++ struct padding/capacity:
    // count (8), one scheme byte per row group, RD widths/dictionary once per
    // group, three uint64 stream offsets per vector, and codec-specific fields.
    // The partial last vector is encoded like the others: no values are dropped.
    size_t comp_bits = stored_metadata.size()*8;
    comp_bits += ffor_stream.size()*64 + exc_pos_stream.size()*16;
    for(const auto& meta:metadata_stream)
        comp_bits += meta.exc_count*(meta.rd ? 16 : 64);
    std::cerr << "ALP vectors=" << alp_vectors << " RD vectors=" << rd_vectors
              << " partial vector values=" << n%VEC << "\n";
    result.access_layout="native_vectors";
    result.compression_unit_values=VEC;
    result.compressed_bits = comp_bits;
    result.compression_ratio = static_cast<double>(result.compressed_bits) / result.uncompressed_bits;
    result.compression_throughput_mbs = (n * sizeof(double) / 1024.0 / 1024.0) / (comp_ns / 1e9);

    // Reopen solely from counted metadata: no fitting state or unrecorded
    // decoder fields may be necessary for a successful round trip. Parsing is
    // outside query timing, just as the other indexed adapters open once.
    metadata_stream.clear();
    rowgroups.clear();
    alp::storer::MemReader reader(stored_metadata.data());
    uint64_t reopened_count=0;
    reader.read(&reopened_count,8);
    if(reopened_count!=n) throw std::runtime_error("ALP metadata count mismatch");
    rowgroups.resize((num_vecs+alp::config::N_VECTORS_PER_ROWGROUP-1)/alp::config::N_VECTORS_PER_ROWGROUP);
    for(auto& st:rowgroups) {
        uint8_t rd=0;
        reader.read(&rd,1);
        st.scheme=rd ? alp::Scheme::ALP_RD : alp::Scheme::ALP;
        if(rd) {
            reader.read(&st.right_bit_width,1);reader.read(&st.left_bit_width,1);
            reader.read(&st.actual_dictionary_size,1);
            reader.read(st.left_parts_dict,st.actual_dictionary_size*2);
        }
    }
    metadata_stream.resize(num_vecs);
    for(size_t v=0;v<num_vecs;++v) {
        auto& m=metadata_stream[v];
        m.rd_group=v/alp::config::N_VECTORS_PER_ROWGROUP;
        m.count=static_cast<uint16_t>(std::min(VEC,n-v*VEC));
        m.rd=rowgroups[m.rd_group].scheme==alp::Scheme::ALP_RD;
        uint64_t offsets[3];reader.read(offsets,sizeof(offsets));
        m.ffor_offset=offsets[0];m.exc_val_offset=offsets[1];m.exc_pos_offset=offsets[2];
        reader.read(&m.exc_count,2);
        if(!m.rd) {
            reader.read(&m.bit_width,1);reader.read(&m.factor,1);reader.read(&m.exponent,1);
            reader.read(&m.ffor_base,8);
        }
    }
    if(reader.get_size()!=stored_metadata.size()) throw std::runtime_error("ALP metadata length mismatch");

    // Full decompression - with checksum to prevent optimization
    std::vector<double> decompressed(num_vecs*VEC); // kernels write whole vectors
    volatile double decomp_checksum = 0;  // volatile prevents optimization
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t v = 0; v < num_vecs; ++v) {
        decode_vector(
            metadata_stream[v],
            ffor_stream.data(),
            exc_val_stream.data(),
            exc_pos_stream.data(), rd_exc_stream.data(),
            decompressed.data() + v * VEC, rowgroups
        );
        // Force actual computation by reading result
        decomp_checksum += decompressed[v * VEC];
    }
    compiler_barrier();  // Compiler barrier before timing ends
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(decompressed);
    decompressed.resize(n);
    do_not_optimize(decomp_checksum);
    auto decomp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.decompression_throughput_mbs = (n * sizeof(double) / 1024.0 / 1024.0) / (decomp_ns / 1e9);

    // Bitwise validation outside timing; retain metrics but flag any failure.
    for(size_t i=0;i<n;++i)
        result.roundtrip_mismatches += std::memcmp(&data[i],&decompressed[i],sizeof(double))!=0;
    result.validation_status=result.roundtrip_mismatches ? "failed" : "passed";
    if(result.roundtrip_mismatches) std::cerr << "Warning: ALP bitwise mismatches=" << result.roundtrip_mismatches << "\n";

    // Random access - with volatile checksum to prevent optimization
    const size_t num_ra_queries = std::min(size_t(10000), bench_data.random_indices.size());
    alignas(64) double vec_buf[VEC];
    volatile double ra_sum = 0;  // volatile prevents optimization

    t1 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < num_ra_queries; ++q) {
        size_t idx = bench_data.random_indices[q];
        size_t vec_idx = idx / VEC;
        size_t offset = idx % VEC;

        decode_vector(
            metadata_stream[vec_idx],
            ffor_stream.data(),
            exc_val_stream.data(),
            exc_pos_stream.data(), rd_exc_stream.data(),
            vec_buf, rowgroups
        );
        ra_sum += vec_buf[offset];
    }
    compiler_barrier();  // Compiler barrier before timing ends
    t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(ra_sum);
    auto ra_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    if (num_ra_queries == 0) {
        result.random_access_ns = 0.0;
        result.random_access_mbs = 0.0;
    } else {
        result.random_access_ns = static_cast<double>(ra_ns) / num_ra_queries;
        result.random_access_mbs = (num_ra_queries * sizeof(double) / 1024.0 / 1024.0) / (ra_ns / 1e9);
    }

    // Random-access correctness spot-check (after timing, to avoid warming caches / predictors).
    // Compare point-decoded values to the original input.
    {
        const size_t check_queries = std::min<size_t>(200, num_ra_queries);
        for (size_t q = 0; q < check_queries; ++q) {
            const size_t idx = bench_data.random_indices[q];
            const size_t vec_idx = idx / VEC;
            const size_t offset = idx % VEC;

            decode_vector(
                metadata_stream[vec_idx],
                ffor_stream.data(),
                exc_val_stream.data(),
                exc_pos_stream.data(), rd_exc_stream.data(),
                vec_buf, rowgroups
            );

            const double got = vec_buf[offset];
            const double expected = data[idx];
            if (std::memcmp(&got,&expected,sizeof(double)) != 0) {
                result.validation_status="failed";
                std::cerr << "ALP random access mismatch at idx=" << idx
                          << " expected=" << expected << " got=" << got << std::endl;
                break;
            }
        }
    }

    // Range queries - with volatile checksum to prevent optimization
    const size_t num_range_queries = 1000;
    for (auto range : range_sizes) {
        if (range >= n) {
            result.range_query_throughputs.emplace_back(range, 0.0);
            continue;
        }

        const auto& range_indices = bench_data.range_query_indices.at(range);
        std::vector<double> range_buf(range);
        volatile double range_checksum = 0;  // volatile prevents optimization

        auto read_range = [&](size_t start_idx) {
            size_t start_vec = start_idx / VEC;
            size_t end_vec = (start_idx + range - 1) / VEC;
            size_t out_pos = 0;

            for (size_t v = start_vec; v <= end_vec && v < num_vecs; ++v) {
                decode_vector(
                    metadata_stream[v],
                    ffor_stream.data(),
                    exc_val_stream.data(),
                    exc_pos_stream.data(), rd_exc_stream.data(),
                    vec_buf, rowgroups
                );

                size_t vec_start_global = v * VEC;
                size_t skip = (v == start_vec) ? (start_idx - vec_start_global) : 0;
                size_t take = std::min(VEC - skip, range - out_pos);

                memcpy(range_buf.data() + out_pos, vec_buf + skip, take * sizeof(double));
                out_pos += take;
            }
        };
        t1 = std::chrono::high_resolution_clock::now();
        size_t q_count=0;
        for(auto start_idx:range_indices) {
            if(q_count>=num_range_queries)break;
            ++q_count;
            read_range(start_idx);
            range_checksum+=range_buf[0];
            do_not_optimize(range_buf);
        }
        compiler_barrier();  // Compiler barrier before timing ends
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(range_checksum);

        auto range_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double throughput = (range * sizeof(double) * q_count / 1024.0 / 1024.0) / (range_ns / 1e9);
        result.range_query_throughputs.emplace_back(range, throughput);
        for(size_t q=0;q<std::min<size_t>(q_count,3);++q) {
            read_range(range_indices[q]);
            if(std::memcmp(range_buf.data(),data.data()+range_indices[q],range*sizeof(double))) {
                result.validation_status="failed";
                std::cerr << "Warning: ALP range mismatch\n";
            }
        }
    }

    return result;
}
