#pragma once
// Elf*: adaptive 17-state Huffman coding, distribution-optimized XOR windows,
// and decimal erasure. See docs/elfstar.md for provenance and exactness guards.
#ifdef __FAST_MATH__
#error "ElfStar requires strict IEEE arithmetic (-fno-fast-math -ffp-contract=off)"
#endif
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gef_compressors { namespace elfstar {
namespace detail {
using Bytes = std::vector<uint8_t>;
constexpr uint64_t end_sign = UINT64_C(0x7ff8000000000000);
inline uint64_t bits(double v) { uint64_t b; std::memcpy(&b,&v,8); return b; }
inline double value(uint64_t b) { double v; std::memcpy(&v,&b,8); return v; }
inline unsigned leading(uint64_t b) { return b ? unsigned(__builtin_clzll(b)) : 64; }
inline unsigned trailing(uint64_t b) { return b ? unsigned(__builtin_ctzll(b)) : 64; }
inline unsigned ceil_log2(unsigned n) { unsigned b=0; for (unsigned x=1;x<n;x<<=1) ++b; return b; }
struct Writer {
    Bytes data;
    size_t position=0;
    void put(uint64_t v,unsigned n) {
        if(n>64) throw std::runtime_error("ElfStar invalid write width");
        while(n) {
            unsigned available=8-unsigned(position%8), take=std::min(available,n);
            if(position/8==data.size()) data.push_back(0);
            data[position/8] |= uint8_t(((v>>(n-take))&((1U<<take)-1))<<(available-take));
            position+=take;n-=take;
        }
    }
};
struct Reader {
    const Bytes& data; size_t position=0;
    explicit Reader(const Bytes& b):data(b) {}
    uint64_t get(unsigned n) {
        if(n>64 || position>data.size()*8 || n>data.size()*8-position)
            throw std::runtime_error("truncated or invalid ElfStar bitstream");
        uint64_t v=0;
        while(n) {
            unsigned available=8-unsigned(position%8), take=std::min(available,n);
            v=(v<<take)|((data[position/8]>>(available-take))&((1U<<take)-1));
            position+=take;n-=take;
        }
        return v;
    }
    void finish() {
        if((position+7)/8!=data.size()) throw std::runtime_error("ElfStar trailing bytes");
        if(position%8 && get(8-unsigned(position%8))) throw std::runtime_error("ElfStar nonzero padding");
    }
};
inline double power10(int exponent) {
    // Java uses decimal parsing, not libm pow(), outside its literal table.
    // strtod provides the same correctly-rounded powers used by the eraser.
    static const auto powers=[] {
        std::array<double,633> p{};
        for(int i=-324;i<=308;++i)p[i+324]=std::strtod(("1e"+std::to_string(i)).c_str(),nullptr);
        return p;
    }();
    if(exponent>308)return std::numeric_limits<double>::infinity();
    if(exponent< -324)return 0;
    return powers[exponent+324];
}
inline int decimal_exponent(double v) {
    if(!(v>0) || !std::isfinite(v)) throw std::runtime_error("invalid erased ElfStar value");
    if(v>=1) {
        double p=10;
        for(int i=0;i<9;++i,p*=10) if(v<p) return i;
    } else {
        for(int i=1;i<=10;++i) if(v>=power10(-i)) return -i;
    }
    return int(std::floor(std::log10(v)));
}
inline double recover(uint64_t raw,int beta) {
    if(beta==16) return value(raw);
    double v=value(raw);
    const int sp=decimal_exponent(std::abs(v));
    if(beta==0) return std::copysign(power10(sp+1),v);
    const int alpha=beta-sp-1;
    if(alpha<0 || alpha>308) throw std::runtime_error("invalid ElfStar alpha");
    const double scale=power10(alpha);
    return (v<0 ? std::floor(v*scale) : std::ceil(v*scale))/scale;
}
struct Erased { uint64_t raw; int beta; };
inline Erased erased(double v,int& last_beta) {
    const uint64_t original=bits(v);
    Erased none{original,16};
    if(v==0 || !std::isfinite(v)) return none;
    const double a=std::abs(v);
    const int sp=decimal_exponent(a);
    bool is_power=false;
    if(a<1 && sp>=-10) is_power=(a==power10(sp));
    else if(a>=1e9 || a<1e-10) {
        double l=std::log10(a); is_power=l==std::floor(l);
    }
    int i;
    if(last_beta!=INT_MAX && last_beta!=0) i=std::max(last_beta-sp-1,1);
    else if(last_beta==INT_MAX) i=17-sp;
    else i=sp>=0 ? 1 : -sp;
    if(i<0 || i>308) return none;
    double scaled=a*power10(i);
    while(std::isfinite(scaled) && scaled<std::ldexp(1.0,63) &&
          scaled!=std::trunc(scaled) && i<308) scaled=a*power10(++i);
    if(!std::isfinite(scaled) || scaled>=std::ldexp(1.0,63) || scaled!=std::trunc(scaled)) return none;
    if(scaled/power10(i)!=a) return none;
    int64_t integer=static_cast<int64_t>(scaled);
    while(i>0 && integer%10==0) { --i;integer/=10; }
    const int significant=sp+i+1, beta=is_power ? 0 : significant;
    const int alpha=significant-sp-1;
    if(beta<0 || beta>=16 || alpha<0) return none;
    static constexpr int f[]={0,4,7,10,14,17,20,24,27,30,34,37,40,44,47,50,54,57,60,64,67};
    int falpha=alpha<21 ? f[alpha] : int(std::ceil(alpha*3.3219280948873626));
    const int e=int((original>>52)&2047);
    const int erase_bits=52-(falpha+e-1023);
    if(erase_bits<=4 || erase_bits>=64) return none;
    uint64_t raw=original&(~UINT64_C(0)<<erase_bits);
    if(raw==original) return none;
    // Extreme/high-precision values sometimes violate upstream's decimal
    // assumptions. Use the existing no-erasure symbol rather than lose bits.
    try { if(bits(recover(raw,beta))!=original) return none; }
    catch(const std::runtime_error&) { return none; }
    last_beta=beta;
    return {raw,beta};
}
struct Code { uint64_t code=0; unsigned length=0; };
using Codes=std::array<Code,17>;
using Distribution=std::array<uint64_t,64>;
using Frequencies=std::array<uint64_t,17>;
struct Node { int symbol=-1,left=-1,right=-1; uint64_t frequency=0; unsigned height=0; };
inline Codes huffman(const Frequencies& frequencies) {
    std::vector<Node> nodes;
    std::vector<int> heap;
    // Java PriorityQueue's exact equal-key behavior is retained for byte-level
    // comparison with the authors' streams; frequency and height are the keys.
    auto less=[&](int a,int b) {return nodes[a].frequency!=nodes[b].frequency ?
        nodes[a].frequency<nodes[b].frequency : nodes[a].height<nodes[b].height;};
    auto push=[&](int x) {
        size_t k=heap.size();heap.push_back(x);
        while(k) {size_t parent=(k-1)/2;if(!less(x,heap[parent])) break;heap[k]=heap[parent];k=parent;}
        heap[k]=x;
    };
    auto pop=[&]() {
        int result=heap.front(),x=heap.back();heap.pop_back();
        if(!heap.empty()) {
            size_t k=0,half=heap.size()/2;
            while(k<half) {size_t child=2*k+1;
                if(child+1<heap.size() && less(heap[child+1],heap[child])) ++child;
                if(!less(heap[child],x)) break;
                heap[k]=heap[child];k=child;
            }
            heap[k]=x;
        }
        return result;
    };
    for(int s=0;s<17;++s) {nodes.push_back({s,-1,-1,frequencies[s],0});push(s);}
    while(heap.size()>1) {
        int l=pop(),r=pop();
        nodes.push_back({-1,l,r,nodes[l].frequency+nodes[r].frequency,1+std::max(nodes[l].height,nodes[r].height)});
        push(int(nodes.size()-1));
    }
    Codes result;
    auto visit=[&](auto&& self,int at,uint64_t bits,unsigned length)->void {
        if(nodes[at].symbol>=0) result[nodes[at].symbol]={bits,length};
        else {self(self,nodes[at].left,bits<<1,length+1);self(self,nodes[at].right,(bits<<1)|1,length+1);}
    };
    visit(visit,heap.front(),0,0);
    return result;
}
inline void write_codes(Writer& w,const Codes& codes) {
    unsigned maxlen=0;for(auto c:codes) maxlen=std::max(maxlen,c.length);
    unsigned width=ceil_log2(maxlen);w.put(width,3);
    for(auto c:codes) {w.put(c.length-1,width);w.put(c.code,c.length);}
}
inline Codes read_codes(Reader& r) {
    unsigned width=unsigned(r.get(3)); if(width>4) throw std::runtime_error("invalid ElfStar Huffman lengths");
    Codes codes;
    for(auto& c:codes) {c.length=unsigned(r.get(width))+1;c.code=r.get(c.length);}
    for(size_t i=0;i<17;++i) for(size_t j=0;j<17;++j) if(i!=j && codes[i].length<=codes[j].length &&
        codes[i].code==(codes[j].code>>(codes[j].length-codes[i].length)))
        throw std::runtime_error("invalid ElfStar Huffman prefix");
    return codes;
}
struct HuffmanReader {
    struct Branch {int next[2]={-1,-1};int symbol=-1;};
    std::vector<Branch> tree{1};
    explicit HuffmanReader(const Codes& codes) {
        for(int symbol=0;symbol<17;++symbol) {
            int at=0;auto c=codes[symbol];
            for(unsigned bit=c.length;bit>0;--bit) {
                unsigned b=unsigned((c.code>>(bit-1))&1);
                int next=tree[at].next[b];
                if(next<0) {next=int(tree.size());tree[at].next[b]=next;tree.push_back({});}
                at=next;
            }
            tree[at].symbol=symbol;
        }
    }
    int read(Reader& r) const {
        int at=0;for(unsigned i=0;i<17;++i) {
            at=tree[at].next[r.get(1)];if(at<0) break;
            if(tree[at].symbol>=0) return tree[at].symbol;
        }
        throw std::runtime_error("invalid ElfStar Huffman code");
    }
};

struct Rule {
    std::vector<unsigned> positions;
    std::array<unsigned,64> round{},representation{};
    unsigned width=0;
    explicit Rule(std::vector<unsigned> p={0}):positions(std::move(p)) {
        if(positions.empty() || positions.size()>32 || positions[0]!=0) throw std::runtime_error("invalid ElfStar rule");
        for(size_t i=1;i<positions.size();++i) if(positions[i]<=positions[i-1] || positions[i]>=64) throw std::runtime_error("invalid ElfStar rule order");
        width=ceil_log2(unsigned(positions.size()));
        size_t at=0;for(unsigned z=0;z<64;++z) {
            if(at+1<positions.size() && positions[at+1]==z) ++at;
            round[z]=positions[at];representation[z]=unsigned(at);
        }
    }
    void write(Writer& w) const {w.put(positions.size(),5);for(auto p:positions)w.put(p,6);}
    static Rule read(Reader& r) {
        unsigned n=unsigned(r.get(5));if(!n)n=32;
        std::vector<unsigned> p(n);for(auto& x:p)x=unsigned(r.get(6));return Rule(std::move(p));
    }
};
inline Rule optimize_rule(const Distribution& a) {
    std::array<int,64> pre{},post{};pre[0]=1;
    uint64_t total=a[0];int nonzero=64;
    for(int i=1;i<64;++i) {total+=a[i];if(!a[i])--nonzero;pre[i]=pre[i-1]+(a[i]!=0);}
    for(int i=0;i<64;++i)post[i]=nonzero-pre[i];
    const unsigned maxz=std::min(ceil_log2(unsigned(nonzero)),5U);
    uint64_t best=UINT64_MAX;std::vector<unsigned> best_positions;
    for(unsigned z=0;z<=maxz;++z) {
        const uint64_t encoding_cost=total*z;if(encoding_cost>=best)break;
        int original=1<<z,num=std::min(original,nonzero);
        uint64_t dp[64][32]{};int predecessor[64][32]{};predecessor[0][0]=-1;
        for(int i=1;i<64;++i) {
            if(!a[i])continue;
            for(int j=std::max(1,num+i-64);j<=i && j<num;++j) {
                if(i>1 && j==1) {for(int k=1;k<i;++k)dp[i][j]+=a[k]*k;predecessor[i][j]=0;}
                else {
                    if(pre[i]<j+1 || post[i]<num-1-j)continue;
                    uint64_t cost=UINT64_MAX;int previous=0;
                    for(int k=j-1;k<i;++k) {
                        if((!a[k]&&k>0)||pre[k]<j||post[k]<num-j)continue;
                        uint64_t sum=dp[k][j-1];for(int p=k+1;p<i;++p)sum+=a[p]*(p-k);
                        if(sum<cost) {cost=sum;previous=k;if(!sum)break;}
                    }
                    if(cost!=UINT64_MAX){dp[i][j]=cost;predecessor[i][j]=previous;}
                }
            }
        }
        uint64_t cost=UINT64_MAX;int last=-1;
        for(int i=num-1;i<64;++i) {
            if(num==1&&i>0)break;
            if((!a[i]&&i>0)||pre[i]<num)continue;
            uint64_t sum=dp[i][num-1];for(int j=i+1;j<64;++j)sum+=a[j]*(j-i);
            if(sum<cost){cost=sum;last=i;}
        }
        std::vector<unsigned> positions(num);
        for(int j=num-1;j>=0;--j){positions[j]=unsigned(last);last=predecessor[last][j];}
        if(original>nonzero) {
            std::vector<unsigned> expanded(original);int j=0,k=0;
            while(j<original&&k<num) {
                if(j-k<original-num&&unsigned(j)<positions[k]){expanded[j]=unsigned(j);++j;}
                else{expanded[j++]=positions[k++];}
            }
            positions=std::move(expanded);
        }
        if(cost+encoding_cost<best){best=cost+encoding_cost;best_positions=std::move(positions);}
    }
    return Rule(std::move(best_positions));
}
struct XorWriter {
    Rule lead,trail;uint64_t previous=0;bool first=true;unsigned stored_lead=65,stored_trail=65;
    XorWriter(Rule l,Rule t):lead(std::move(l)),trail(std::move(t)) {}
    void add(Writer& w,uint64_t raw) {
        if(first){first=false;lead.write(w);trail.write(w);unsigned tz=trailing(raw);w.put(tz,7);
            if(tz<63)w.put(raw>>(tz+1),63-tz);previous=raw;return;}
        uint64_t x=previous^raw;
        if(!x){w.put(1,2);return;}
        unsigned l=lead.round[leading(x)],t=trail.round[trailing(x)];
        if(l>=stored_lead && t>=stored_trail && l-stored_lead+t-stored_trail<1+lead.width+trail.width){
            w.put(1,1);w.put(x>>stored_trail,64-stored_lead-stored_trail);
        }else{
            stored_lead=l;stored_trail=t;w.put(0,2);w.put(lead.representation[l],lead.width);w.put(trail.representation[t],trail.width);w.put(x>>t,64-l-t);
        }
        previous=raw;
    }
};
struct XorReader {
    Rule lead,trail;uint64_t previous=0;bool first=true;unsigned stored_lead=65,stored_trail=65;
    uint64_t next(Reader& r) {
        if(first){first=false;lead=Rule::read(r);trail=Rule::read(r);unsigned tz=unsigned(r.get(7));
            if(tz>64)throw std::runtime_error("invalid ElfStar first value");
            previous=tz==64?0:((r.get(63-tz)<<1)|1)<<tz;return previous;}
        if(!r.get(1)){
            if(r.get(1))return previous;
            unsigned l=unsigned(r.get(lead.width)),t=unsigned(r.get(trail.width));
            if(l>=lead.positions.size()||t>=trail.positions.size())throw std::runtime_error("invalid ElfStar window index");
            stored_lead=lead.positions[l];stored_trail=trail.positions[t];
        }
        if(stored_lead+stored_trail>=64)throw std::runtime_error("invalid ElfStar shared window");
        previous^=r.get(64-stored_lead-stored_trail)<<stored_trail;return previous;
    }
};
} // detail
inline std::vector<uint8_t> compress(const std::vector<double>& values) {
    using namespace detail;
    std::vector<Erased> erased_values;erased_values.reserve(values.size());
    Frequencies frequencies{};Distribution leads{},trails{};int last_beta=INT_MAX;
    uint64_t previous=0;bool first=true;
    for(double v:values){auto e=erased(v,last_beta);erased_values.push_back(e);++frequencies[e.beta];
        if(!first){uint64_t x=previous^e.raw;if(x){++leads[leading(x)];++trails[trailing(x)];}}
        first=false;previous=e.raw;}
    Writer w;w.put(values.size(),64);
    auto codes=huffman(frequencies);write_codes(w,codes);
    XorWriter xor_writer(optimize_rule(leads),optimize_rule(trails));
    for(auto e:erased_values){w.put(codes[e.beta].code,codes[e.beta].length);xor_writer.add(w,e.raw);}
    w.put(codes[16].code,codes[16].length);xor_writer.add(w,end_sign);
    return std::move(w.data);
}
inline std::vector<double> decompress(const std::vector<uint8_t>& bytes,size_t count) {
    using namespace detail;
    Reader r(bytes);if(r.get(64)!=count)throw std::runtime_error("ElfStar count mismatch");
    if(count>bytes.size()*8)throw std::runtime_error("impossible ElfStar count");
    auto codes=read_codes(r);HuffmanReader h(codes);XorReader x;
    std::vector<double> result;result.reserve(count);
    for(size_t i=0;i<count;++i){int beta=h.read(r);result.push_back(recover(x.next(r),beta));}
    if(h.read(r)!=16 || x.next(r)!=end_sign)throw std::runtime_error("invalid ElfStar end marker");
    r.finish();return result;
}
// Decodes only the requested prefix from the beginning of this block. A prefix
// read validates consumed metadata/data, not the unread suffix or end marker.
inline std::vector<double> decompress_prefix(const std::vector<uint8_t>& bytes,
                                             size_t total_count,size_t prefix_count) {
    using namespace detail;
    if(prefix_count>total_count)throw std::invalid_argument("ElfStar prefix exceeds total count");
    Reader r(bytes);if(r.get(64)!=total_count)throw std::runtime_error("ElfStar count mismatch");
    if(total_count>bytes.size()*8)throw std::runtime_error("impossible ElfStar count");
    auto codes=read_codes(r);HuffmanReader h(codes);XorReader x;
    std::vector<double> result;result.reserve(prefix_count);
    if(!prefix_count) {
        // The first Huffman symbol precedes the two XOR rule tables on wire.
        // Validate those tables even when no values were requested.
        h.read(r);Rule::read(r);Rule::read(r);
        return result;
    }
    for(size_t i=0;i<prefix_count;++i){int beta=h.read(r);result.push_back(recover(x.next(r),beta));}
    return result;
}
}} // namespace gef_compressors::elfstar
