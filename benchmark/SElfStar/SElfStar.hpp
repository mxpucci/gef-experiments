#pragma once
// SElf*: streaming state from the preceding window, with explicit decoder
// checkpoints for independently indexed blocks. See docs/selfstar.md.
#include "../ElfStar/ElfStar.hpp"

namespace gef_compressors { namespace selfstar {
namespace detail {
namespace shared = gef_compressors::elfstar::detail;
using shared::Bytes;
using shared::Writer;
using shared::Reader;
using shared::Rule;
using shared::Codes;
using shared::Frequencies;
using shared::Distribution;
constexpr uint32_t magic = UINT32_C(0x53455331); // SES1

struct State {
    bool first_block=true;
    Rule lead{std::vector<unsigned>{0,8,12,16,18,20,22,24}};
    Rule trail{std::vector<unsigned>{0,22,28,32,36,40,42,46}};
    Codes codes{};
    unsigned stored_lead=65,stored_trail=65;
};
inline void write_checkpoint(Writer& w,const State& s,size_t count) {
    w.put(magic,32);w.put(count,64);w.put(s.first_block,1);
    w.put(s.stored_lead,7);w.put(s.stored_trail,7);
    s.lead.write(w);s.trail.write(w);
    if(!s.first_block) shared::write_codes(w,s.codes);
    if(w.position%8) w.put(0,8-unsigned(w.position%8));
}
inline State read_checkpoint(Reader& r,size_t count) {
    if(r.get(32)!=magic || r.get(64)!=count) throw std::runtime_error("SElfStar header/count mismatch");
    if(count>r.data.size()*8) throw std::runtime_error("impossible SElfStar value count");
    State s;s.first_block=bool(r.get(1));
    s.stored_lead=unsigned(r.get(7));s.stored_trail=unsigned(r.get(7));
    if(!((s.stored_lead==65&&s.stored_trail==65) || s.stored_lead+s.stored_trail<64))
        throw std::runtime_error("invalid SElfStar checkpoint window");
    s.lead=Rule::read(r);s.trail=Rule::read(r);
    if(!s.first_block)s.codes=shared::read_codes(r);
    if(r.position%8 && r.get(8-unsigned(r.position%8)))
        throw std::runtime_error("invalid SElfStar checkpoint padding");
    return s;
}
struct XorWriter {
    State& state; bool write_positions;
    bool first=true;uint64_t previous=0;
    Distribution leads{},trails{};
    XorWriter(State& s,bool positions):state(s),write_positions(positions){}
    void add(Writer& w,uint64_t raw) {
        if(first) {
            first=false;w.put(write_positions,1);
            if(write_positions){state.lead.write(w);state.trail.write(w);}
            unsigned tz=shared::trailing(raw);w.put(tz,7);
            if(tz<63)w.put(raw>>(tz+1),63-tz);
            previous=raw;return;
        }
        const uint64_t x=previous^raw;
        if(!x){w.put(1,2);return;}
        unsigned l=shared::leading(x),t=shared::trailing(x);
        ++leads[l];++trails[t];l=state.lead.round[l];t=state.trail.round[t];
        if(l>=state.stored_lead && t>=state.stored_trail &&
            l-state.stored_lead+t-state.stored_trail<1+state.lead.width+state.trail.width) {
            w.put(1,1);w.put(x>>state.stored_trail,64-state.stored_lead-state.stored_trail);
        } else {
            state.stored_lead=l;state.stored_trail=t;
            w.put(0,2);w.put(state.lead.representation[l],state.lead.width);
            w.put(state.trail.representation[t],state.trail.width);w.put(x>>t,64-l-t);
        }
        previous=raw;
    }
};
struct XorReader {
    State& state;bool first=true;uint64_t previous=0;
    explicit XorReader(State& s):state(s){}
    uint64_t next(Reader& r) {
        if(first) {
            first=false;
            if(r.get(1)){state.lead=Rule::read(r);state.trail=Rule::read(r);}
            unsigned tz=unsigned(r.get(7));
            if(tz>64)throw std::runtime_error("invalid SElfStar first value");
            previous=tz==64 ? 0 : ((r.get(63-tz)<<1)|1)<<tz;
            return previous;
        }
        if(!r.get(1)) {
            if(r.get(1))return previous;
            unsigned l=unsigned(r.get(state.lead.width)),t=unsigned(r.get(state.trail.width));
            if(l>=state.lead.positions.size()||t>=state.trail.positions.size())
                throw std::runtime_error("invalid SElfStar table index");
            state.stored_lead=state.lead.positions[l];state.stored_trail=state.trail.positions[t];
        }
        if(state.stored_lead+state.stored_trail>=64)throw std::runtime_error("invalid SElfStar shared window");
        previous^=r.get(64-state.stored_lead-state.stored_trail)<<state.stored_trail;
        return previous;
    }
};
} // namespace detail

// One Encoder per series. Consecutive calls are consecutive adaptive windows.
// Each returned block includes enough decoder state to decode in isolation.
class Encoder {
    detail::State state_;
    bool write_positions_=false;
    double previous_ratio_=0;
public:
    std::vector<uint8_t> compress_block(const std::vector<double>& values) {
        using namespace detail;
        Writer checkpoint;write_checkpoint(checkpoint,state_,values.size());
        Writer payload;XorWriter x(state_,write_positions_);
        Frequencies frequency{};int last_beta=INT_MAX;
        for(double v:values) {
            const int previous_beta=last_beta;
            auto e=shared::erased(v,last_beta);
            ++frequency[e.beta];
            if(state_.first_block) {
                if(e.beta==16)payload.put(2,2);
                else if(e.beta==previous_beta)payload.put(0,1);
                else{payload.put(3,2);payload.put(e.beta,4);}
            } else payload.put(state_.codes[e.beta].code,state_.codes[e.beta].length);
            x.add(payload,e.raw);
        }
        // Upstream computes the adaptation trigger before the end marker and
        // excludes checkpoint/index overhead from this internal decision.
        const double ratio=values.empty()?0:double(payload.position)/(values.size()*64.0);
        const bool update=previous_ratio_<ratio;
        previous_ratio_=ratio;
        if(state_.first_block)payload.put(2,2);
        else payload.put(state_.codes[16].code,state_.codes[16].length);
        x.add(payload,shared::end_sign);
        state_.codes=shared::huffman(frequency);
        state_.first_block=false;
        if(update){state_.lead=shared::optimize_rule(x.leads);state_.trail=shared::optimize_rule(x.trails);}
        write_positions_=update;
        checkpoint.data.insert(checkpoint.data.end(),payload.data.begin(),payload.data.end());
        return std::move(checkpoint.data);
    }
};

// Convenience cold-start block. For benchmarking multiple windows use Encoder.
inline std::vector<uint8_t> compress(const std::vector<double>& values) {
    Encoder encoder;return encoder.compress_block(values);
}
namespace detail {
inline std::vector<double> decode(const std::vector<uint8_t>& bytes,size_t count,
                                  size_t prefix_count,bool validate_end) {
    if(prefix_count>count)throw std::invalid_argument("SElfStar prefix exceeds total count");
    Reader r(bytes);State state=read_checkpoint(r,count);XorReader x(state);
    // The first window uses Elf+'s prefix states. Later windows use the preceding
    // window's Huffman tree, recovered here from the charged checkpoint.
    std::vector<double> result;result.reserve(prefix_count);int last_beta=INT_MAX;
    if(!prefix_count && !validate_end)return result;
    shared::HuffmanReader h(state.first_block?shared::huffman(Frequencies{}):state.codes);
    auto symbol=[&](){
        if(!state.first_block)return h.read(r);
        if(!r.get(1)) {
            if(last_beta==INT_MAX)throw std::runtime_error("SElfStar missing previous beta");
            return last_beta;
        }
        if(!r.get(1))return 16;
        last_beta=int(r.get(4));return last_beta;
    };
    for(size_t i=0;i<prefix_count;++i){int beta=symbol();result.push_back(shared::recover(x.next(r),beta));}
    if(validate_end) {
        if(symbol()!=16 || x.next(r)!=shared::end_sign)throw std::runtime_error("invalid SElfStar end marker");
        r.finish();
    }
    return result;
}
} // namespace detail
inline std::vector<double> decompress(const std::vector<uint8_t>& bytes,size_t count) {
    return detail::decode(bytes,count,count,true);
}
// Restore the charged checkpoint and stop immediately after the requested
// prefix. No preceding blocks or values beyond prefix_count are decoded.
inline std::vector<double> decompress_prefix(const std::vector<uint8_t>& bytes,
                                             size_t total_count,size_t prefix_count) {
    return detail::decode(bytes,total_count,prefix_count,prefix_count==total_count);
}
}} // namespace gef_compressors::selfstar
