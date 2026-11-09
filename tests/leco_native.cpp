#include "benchmark_common.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
static void check(std::string name,std::vector<int64_t> data){
 BenchmarkData b;b.filename=name;b.min_val=*std::min_element(data.begin(),data.end());
 for(auto&v:data)v-=b.min_val;b.shifted_data=std::move(data);size_t n=b.shifted_data.size();b.uncompressed_bits=n*64;
 for(size_t i=0;i<1000;++i)b.random_indices.push_back((i*7919)%n);
 for(size_t r:{size_t(3),size_t(100),size_t(1000)})for(size_t i=0;i<1000;++i)b.range_query_indices[r].push_back((i*7919)%(n-r));
 auto x=benchmark_leco(b,{3,100,1000});
 if(x.validation_status!="passed"||x.roundtrip_mismatches||x.compression_unit_values==1000||x.access_layout!="native_fixed_partitions_tuned")throw std::runtime_error("LeCo validation failed: "+name);
 std::cout<<name<<" n="<<n<<" native_partition="<<x.compression_unit_values<<" bits="<<x.compressed_bits<<" validation="<<x.validation_status<<'\n';
}
int main(int argc,char**argv){
 std::vector<int64_t> v;for(int64_t i=0;i<50123;++i)v.push_back((i/1000)*100000+(i%1000)*7);check("synthetic",v);
 for(int k=1;k<argc;++k){std::ifstream f(argv[k],std::ios::binary);uint64_t n,d;f.read((char*)&n,8);f.read((char*)&d,8);v.resize(std::min<uint64_t>(n,30000));f.read((char*)v.data(),v.size()*8);check(argv[k],v);}
}
