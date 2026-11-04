#include "/home/testPF/gef-experiments/benchmark/benchmark_common.hpp"
#include <fstream>
int main(int argc,char**argv){for(int f=1;f<argc;f++){std::ifstream in(argv[f],std::ios::binary);BenchmarkData b;uint64_t n;in.read((char*)&n,8);in.read((char*)&b.decimals,8);n=std::min<uint64_t>(n,10000);b.shifted_data.resize(n);in.read((char*)b.shifted_data.data(),n*8);b.convert_shifted_to_double();std::string s=argv[f];auto p=s.find_last_of('/');std::ofstream out("/tmp/gef-elf-audit-"+s.substr(p+1)+".f64",std::ios::binary);out.write((char*)b.double_data.data(),n*8);}}
