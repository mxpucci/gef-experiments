#include "../benchmark/alp_benchmark.cpp"
#include <random>
#include <fstream>
#include <stdexcept>
static void require(bool ok) { if(!ok) throw std::runtime_error("ALP adapter regression"); }
static void check(BenchmarkData& d) {
    const size_t n=d.double_data.size();
        d.random_indices={0,n/2,n-1};
        for(size_t boundary: {size_t(1024),size_t(102400)})
            if(boundary<n) {d.random_indices.push_back(boundary-1);d.random_indices.push_back(boundary);}
        std::vector<size_t> ranges;
        if(n>2) {ranges={2};d.range_query_indices[2]={0,n-2};
            for(size_t boundary: {size_t(1024),size_t(102400)})
                if(boundary<n)d.range_query_indices[2].push_back(boundary-1);
        }
        if(n>2048){ranges.push_back(2048);d.range_query_indices[2048]={0,n-2048};}
        auto result=benchmark_alp(d,ranges);
        require(result.validation_status=="passed" && result.roundtrip_mismatches==0);
        require(result.num_values==n);
        // Independently call upstream encoding to compute
        // its payload lengths; check the adapter's explicitly counted metadata.
        // Partial last vector: exceptions are counted by a scalar check over
        // its real values only, and the packed size uses the FastLanes prefix.
        size_t expected=8;
        auto prefix_bytes=[](size_t m,unsigned bw,unsigned w){size_t lanes=1024/w,rows=(m+lanes-1)/lanes;
            return lanes*((rows*bw+w-1)/w)*w/8;};
        std::vector<double> sample(alp::config::ROWGROUP_SIZE);
        for(size_t base=0;base<n;base+=alp::config::ROWGROUP_SIZE) {
            alp::state<double> state;
            alp::encoder<double>::init(d.double_data.data(),base,n,sample.data(),state);
            ++expected;
            bool rd=state.scheme==alp::Scheme::ALP_RD;
            if(rd) {alp::rd_encoder<double>::init(d.double_data.data(),base,n,sample.data(),state);
                expected+=3+state.actual_dictionary_size*2;}
            for(size_t v=base;v<std::min(base+alp::config::ROWGROUP_SIZE,n);v+=1024) {
                const size_t m=std::min<size_t>(1024,n-v);
                double in[1024];std::copy_n(d.double_data.data()+v,m,in);std::fill(in+m,in+1024,in[m-1]);
                if(m<1024)state.vector_size=uint16_t(m);
                uint16_t count=0,pos[1024],ex[1024],left[1024];
                uint64_t right[1024];int64_t encoded[1024],for_base;double exc[1024];uint8_t width;
                if(rd) {alp::rd_encoder<double>::encode(in,ex,pos,&count,right,left,state);
                    size_t real=0;for(size_t i=0;i<m;++i){uint64_t b;std::memcpy(&b,&in[i],8);
                        real+=state.left_parts_dict_map.count(uint16_t(b>>state.right_bit_width))==0;}
                    require(real<=count && (m<1024 || real==count));
                    expected+=24+2+prefix_bytes(m,state.right_bit_width,64)+prefix_bytes(m,state.left_bit_width,16)+real*4;}
                else {alp::encoder<double>::encode(in,exc,pos,&count,encoded,state);
                    alp::encoder<double>::analyze_ffor(encoded,width,&for_base);
                    int64_t lo=INT64_MAX,hi=INT64_MIN;size_t real=0;
                    for(size_t i=0;i<m;++i){lo=std::min(lo,encoded[i]);hi=std::max(hi,encoded[i]);}
                    for(size_t i=0;i<count;++i)real+=pos[i]<m;
                    // padding must not widen the frame of reference
                    require(lo==for_base && uint8_t(width)==alp::encoder<double>::count_bits(hi,lo));
                    expected+=24+13+prefix_bytes(m,width,64)+real*10;}
            }
        }
        require(result.compressed_bits==expected*8);
        std::cout << d.filename << "," << n << "," << result.compressed_bits << ",passed\n";
}
int main(int argc,char** argv) {
    for(int mode=0;mode<4;++mode) for(size_t n: {size_t(1),size_t(31),size_t(1023),size_t(1024),size_t(1500),size_t(2049),size_t(7000),size_t(102400),size_t(103425),size_t(204800)}) {
        BenchmarkData d{};
        d.double_data.resize(n); d.uncompressed_bits=n*64;
        std::mt19937_64 rng(42);
        for(size_t i=0;i<n;++i) {
            if(mode==0)d.double_data[i]=double(int(i%10000)-5000)/100.0;
            else if(mode==1) { auto bits=rng();std::memcpy(&d.double_data[i],&bits,8); }
            else if(mode==2) d.double_data[i]=3.0;
            else if(i/alp::config::ROWGROUP_SIZE%2==0)d.double_data[i]=double(i%10000)/100.0;
            else {auto bits=rng();std::memcpy(&d.double_data[i],&bits,8);}
        }
        d.filename="synthetic-"+std::to_string(mode);
        check(d);
    }
    for(int arg=1;arg<argc;++arg) {
        std::ifstream input(argv[arg],std::ios::binary|std::ios::ate);
        require(bool(input));
        const auto bytes=static_cast<size_t>(input.tellg());
        require(bytes>=16 && (bytes-16)%8==0);
        input.seekg(0);uint64_t original_count=0,decimals=0;
        input.read(reinterpret_cast<char*>(&original_count),8);
        input.read(reinterpret_cast<char*>(&decimals),8);
        require(decimals<=18);
        const size_t n=(bytes-16)/8;require(n>0 && n<=original_count);
        BenchmarkData d{};d.filename=argv[arg];d.uncompressed_bits=n*64;
        int64_t scale=1;for(size_t i=0;i<decimals;++i)scale*=10;
        for(size_t i=0;i<n;++i){int64_t raw;input.read(reinterpret_cast<char*>(&raw),8);
            require(bool(input));d.double_data.push_back(static_cast<double>(raw)/static_cast<double>(scale));}
        check(d);
    }
}
