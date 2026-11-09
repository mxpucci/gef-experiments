#include "benchmark_common.hpp"
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

BenchmarkResult benchmark_camel(const BenchmarkData& data,const std::vector<size_t>& ranges) {
    BenchmarkResult r; r.compressor="Camel";r.dataset=data.filename;
    r.num_values=data.double_data.size();r.uncompressed_bits=data.uncompressed_bits;
    r.access_layout="original_java_independent_blocks";r.compression_unit_values=1000;
    struct Temporary {
        std::filesystem::path dir;
        ~Temporary(){if(!dir.empty()){std::error_code ec;std::filesystem::remove_all(dir,ec);}}
    } temp;
    try {
        std::string cp;
        if(const char* env=std::getenv("GEF_CAMEL_CLASSPATH"))cp=env;
        else {std::ifstream f(NEATS_CAMEL_CLASSPATH_FILE);std::getline(f,cp);}
        if(cp.empty())throw std::runtime_error("original Camel is not built; run tools/build_camel_java.py");
        auto pattern=(std::filesystem::temp_directory_path()/"gef-camel-XXXXXX").string();
        char* name=mkdtemp(pattern.data());if(!name)throw std::runtime_error("cannot create Camel input directory");temp.dir=name;
        auto input=temp.dir/"input.bin",output=temp.dir/"metrics.txt";
        std::ofstream f(input,std::ios::binary);
        auto put=[&](uint64_t v){for(unsigned i=0;i<8;i++)f.put(static_cast<char>(v>>(i*8)));};
        put(data.double_data.size());for(double v:data.double_data)put(std::bit_cast<uint64_t>(v));
        size_t q=std::min<size_t>(100000,data.random_indices.size());put(q);for(size_t i=0;i<q;i++)put(data.random_indices[i]);
        size_t count=0;for(size_t range:ranges)if(range<r.num_values)++count;put(count);
        for(size_t range:ranges)if(range<r.num_values){const auto& starts=data.range_query_indices.at(range);size_t n=std::min<size_t>(1000,starts.size());put(range);put(n);for(size_t i=0;i<n;i++)put(starts[i]);}
        f.close();if(!f)throw std::runtime_error("failed writing Camel input");
        // No shell. JVM startup and IPC are outside the Java codec timers.
        pid_t child=fork();if(child<0)throw std::runtime_error("cannot start Camel JVM");
        if(child==0){execlp("java","java","-Xmx32g","-cp",cp.c_str(),"CamelBenchmark",input.c_str(),output.c_str(),static_cast<char*>(nullptr));_exit(127);}
        int status=0;if(waitpid(child,&status,0)<0 || !WIFEXITED(status) || WEXITSTATUS(status)!=0)throw std::runtime_error("original Camel JVM failed");
        std::ifstream metrics(output);uint64_t comp,dec,ra,queries,range_count;
        if(!(metrics>>r.compressed_bits>>comp>>dec>>r.roundtrip_mismatches>>ra>>queries>>range_count))throw std::runtime_error("missing Camel measurements");
        auto mibs=[](double bytes,uint64_t ns){return ns?bytes/1048576.0/(double(ns)/1e9):0.0;};
        r.compression_ratio=double(r.compressed_bits)/r.uncompressed_bits;
        r.compression_throughput_mbs=mibs(r.num_values*8.0,comp);r.decompression_throughput_mbs=mibs(r.num_values*8.0,dec);
        r.random_access_ns=queries?double(ra)/queries:0;r.random_access_mbs=mibs(queries*8.0,ra);
        for(uint64_t k=0;k<range_count;k++){uint64_t range,ns,n;if(!(metrics>>range>>ns>>n))throw std::runtime_error("incomplete Camel range measurements");r.range_query_throughputs.emplace_back(range,mibs(range*8.0*n,ns));}
        r.validation_status=r.roundtrip_mismatches?"bitwise_failed":"passed";
        if(r.roundtrip_mismatches)std::cerr<<"WARNING: original Camel changed "<<r.roundtrip_mismatches<<" values; results are not lossless for this dataset\n";
    } catch(const std::exception& e) {
        r.validation_status="error";r.compression_ratio=std::numeric_limits<double>::quiet_NaN();
        std::cerr<<"WARNING: Camel: "<<e.what()<<'\n';
    }
    return r;
}
