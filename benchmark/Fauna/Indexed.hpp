#pragma once
#include "Fauna.hpp"
#include <utility>
extern "C" {
int gef_fauna_indexed_compress(const double*,std::size_t,std::size_t,void**);
int gef_fauna_indexed_open(const std::uint8_t*,std::size_t,void**);
int gef_fauna_indexed_info(const void*,std::size_t*,const std::uint8_t**,std::size_t*);
int gef_fauna_indexed_read(const void*,std::size_t,std::size_t,double*);
void gef_fauna_indexed_free(void*);
}
namespace gef_compressors { namespace fauna {
class Indexed {
    void* handle_=nullptr;
    std::size_t n_=0,bytes_=0;
    const std::uint8_t* data_=nullptr;
    void info(){fauna_detail::check(gef_fauna_indexed_info(handle_,&n_,&data_,&bytes_));}
public:
    explicit Indexed(const std::vector<double>& input,std::size_t batch=65536){
        fauna_detail::check(gef_fauna_indexed_compress(input.data(),input.size(),batch,&handle_));info();
    }
    explicit Indexed(const std::vector<std::uint8_t>& bytes){
        fauna_detail::check(gef_fauna_indexed_open(bytes.data(),bytes.size(),&handle_));info();
    }
    ~Indexed(){gef_fauna_indexed_free(handle_);}
    Indexed(const Indexed&)=delete;
    Indexed& operator=(const Indexed&)=delete;
    Indexed(Indexed&& other) noexcept:handle_(std::exchange(other.handle_,nullptr)),n_(other.n_),bytes_(other.bytes_),data_(other.data_){}
    std::size_t size() const{return n_;}
    std::size_t size_in_bytes() const{return bytes_;}
    std::size_t changed_values() const{return 0;}  // f64 input is encoded unchanged
    std::vector<std::uint8_t> bytes() const{return {data_,data_+bytes_};}
    void read(std::size_t start,std::size_t count,double* output) const{
        fauna_detail::check(gef_fauna_indexed_read(handle_,start,count,output));
    }
};
}}
