#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::cout << "dataset,num_values,decimal_places,changed_f64_values,changed_decimal_values,first_changed_index,original_integer,original_f64,narrowed_f32,recovered_integer\n";
  std::cout << std::setprecision(17);
  for (int a=1; a<argc; ++a) {
    std::ifstream in(argv[a],std::ios::binary);
    uint64_t n=0, decimals=0;
    in.read(reinterpret_cast<char*>(&n),8);
    in.read(reinterpret_cast<char*>(&decimals),8);
    if (!in || decimals>15) throw std::runtime_error("invalid header");
    int64_t scale_int=1;
    for (uint64_t j=0;j<decimals;++j) scale_int*=10;
    const double scale=static_cast<double>(scale_int);
    uint64_t changed_binary=0,changed_decimal=0,first=0;
    int64_t first_raw=0,first_recovered=0;
    double first_double=0,first_float=0;
    std::vector<int64_t> values(1<<20);
    for (uint64_t start=0; start<n;) {
      size_t count=static_cast<size_t>(std::min<uint64_t>(values.size(),n-start));
      in.read(reinterpret_cast<char*>(values.data()),count*8);
      if (!in) throw std::runtime_error("truncated input");
      for (size_t j=0;j<count;++j) {
        double value=static_cast<double>(values[j])/scale;
        float narrowed=static_cast<float>(value);
        double expanded=static_cast<double>(narrowed);
        changed_binary += value!=expanded;
        int64_t recovered=std::llround(expanded*scale);
        if (recovered!=values[j]) {
          if (!changed_decimal) {
            first=start+j;first_raw=values[j];first_recovered=recovered;
            first_double=value;first_float=expanded;
          }
          ++changed_decimal;
        }
      }
      start+=count;
    }
    std::string name(argv[a]);
    name=name.substr(name.find_last_of('/')+1);
    name=name.substr(0,name.find_last_of('.'));
    std::cout << name << ',' << n << ',' << decimals << ',' << changed_binary << ',' << changed_decimal;
    if (changed_decimal) std::cout << ',' << first << ',' << first_raw << ',' << first_double << ',' << first_float << ',' << first_recovered;
    else std::cout << ",,,,,";
    std::cout << std::endl;
  }
}
