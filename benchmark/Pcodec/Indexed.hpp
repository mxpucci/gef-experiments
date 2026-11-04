#pragma once
#include "Pcodec.hpp"
#include <utility>

extern "C" {
int gef_pco_indexed_compress_f64(const double*,std::size_t,std::size_t,std::size_t,void**);
int gef_pco_indexed_compress_i64(const std::int64_t*,std::size_t,std::size_t,std::size_t,void**);
int gef_pco_indexed_read_f64(const void*,std::size_t,std::size_t,double*);
int gef_pco_indexed_read_i64(const void*,std::size_t,std::size_t,std::int64_t*);
int gef_pco_indexed_open(const std::uint8_t*,std::size_t,std::size_t,void**);
int gef_pco_indexed_info(const void*,std::size_t*,const std::uint8_t**,std::size_t*);
void gef_pco_indexed_free(void*);
}
namespace gef_compressors { namespace pcodec {
namespace detail {
template<class T> struct IndexedAPI;
template<> struct IndexedAPI<double> {
    static constexpr std::size_t kind=1;
    static constexpr auto encode=gef_pco_indexed_compress_f64;
    static constexpr auto read=gef_pco_indexed_read_f64;
};
template<> struct IndexedAPI<std::int64_t> {
    static constexpr std::size_t kind=2;
    static constexpr auto encode=gef_pco_indexed_compress_i64;
    static constexpr auto read=gef_pco_indexed_read_i64;
};
}
template<class T> class Indexed {
    void* handle_ = nullptr;
    std::size_t n_=0, bytes_=0;
    const std::uint8_t* data_=nullptr;
    void info() { detail::check(gef_pco_indexed_info(handle_,&n_,&data_,&bytes_)); }
public:
    Indexed(const std::vector<T>& input,std::size_t chunk=262144,std::size_t page=262144) {
        detail::check(detail::IndexedAPI<T>::encode(input.data(),input.size(),chunk,page,&handle_));
        info();
    }
    explicit Indexed(const std::vector<std::uint8_t>& bytes) {
        detail::check(gef_pco_indexed_open(bytes.data(),bytes.size(),detail::IndexedAPI<T>::kind,&handle_));
        info();
    }
    ~Indexed(){gef_pco_indexed_free(handle_);}
    Indexed(const Indexed&)=delete;
    Indexed& operator=(const Indexed&)=delete;
    Indexed(Indexed&& other) noexcept:handle_(std::exchange(other.handle_,nullptr)),n_(other.n_),bytes_(other.bytes_),data_(other.data_){}
    std::size_t size() const {return n_;}
    std::size_t size_in_bytes() const {return bytes_;}
    std::size_t changed_values() const {return 0;}
    std::vector<std::uint8_t> bytes() const {return {data_,data_+bytes_};}
    void read(std::size_t start,std::size_t count,T* output) const {
        detail::check(detail::IndexedAPI<T>::read(handle_,start,count,output));
    }
};
}}
