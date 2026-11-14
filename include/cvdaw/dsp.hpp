#pragma once
#include "ipc.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
namespace cvdaw {
class Scheduler {
 std::array<Command,1024> heap_{};unsigned size_=0;
 static bool less(const Command&a,const Command&b){return a.frame<b.frame;}
public:
 bool add(Command c)noexcept {
  if(size_==heap_.size())return false;
  unsigned i=size_++;heap_[i]=c;
  while(i){unsigned p=(i-1)/2;if(!less(heap_[i],heap_[p]))break;std::swap(heap_[i],heap_[p]);i=p;}return true;
 }
 bool due(uint64_t frame,Command& c)noexcept {
  if(!size_||heap_[0].frame>frame)return false;
  c=heap_[0];heap_[0]=heap_[--size_];unsigned i=0;
  while(2*i+1<size_){unsigned j=2*i+1;if(j+1<size_ && less(heap_[j+1],heap_[j]))++j;
   if(!less(heap_[j],heap_[i]))break;
   std::swap(heap_[i],heap_[j]);i=j;
  }return true;
 }
};
struct Voice {float phase{},increment{},gain{};uint32_t age{};bool active=false;uint8_t preset{};};
struct Svf {
 float ic1{},ic2{},g=.1f;
 void cutoff(float hz)noexcept{g=std::tan(float(std::numbers::pi)*std::clamp(hz,40.f,16000.f)/rate);}
 float step(float x)noexcept{
  constexpr float k=1.41421356f;float a=1/(1+g*(g+k));
  float v1=a*(ic1+g*(x-ic2)),v2=ic2+g*v1;ic1=2*v1-ic1;ic2=2*v2-ic2;
  if(std::abs(ic1)<1e-20f)ic1=0;
  if(std::abs(ic2)<1e-20f)ic2=0;
  return v2;
 }
};
class Engine {
 std::array<std::array<Voice,8>,4> voices_{};
 std::array<std::array<float,2048>,2> tables_{};
 std::array<float,24000> drum_{};
 std::array<Svf,4> filters_{};
 std::array<uint8_t,4> preset_{};std::array<bool,4> muted_{};
 Scheduler scheduler_;
 uint64_t frame_=0;
public:
 Engine(){
  preset_[2]=preset_[3]=2;
  uint32_t rng=1234567;
  for(unsigned i=0;i<2048;++i){tables_[0][i]=std::sin(2*std::numbers::pi*i/2048);tables_[1][i]=2.f*i/2048-1;}
  // Startup-built RAM kick sample; can replace via load_drum before RT activation.
  double phase=0;
  for(unsigned i=0;i<drum_.size();++i){double t=double(i)/rate;phase+=2*std::numbers::pi*(45+110*std::exp(-40*t))/rate;
   rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
   drum_[i]=float(.8*std::sin(phase)*std::exp(-12*t)+.12*(double(rng)/UINT32_MAX*2-1)*std::exp(-100*t));}
 }
 bool load_drum(const float* p,size_t n){if(n>drum_.size())return false;drum_.fill(0);std::copy_n(p,n,drum_.begin());return true;}
 uint64_t frame()const{return frame_;}
 void command(Command c)noexcept{
  if(c.op==Op::panic){for(auto& bank:voices_)for(auto& v:bank)v.active=false;return;}
  if(c.limb>=4)return;
  if(c.op==Op::mute){muted_[c.limb]=c.value!=0;for(auto& v:voices_[c.limb])v.active=false;return;}
  if(c.op==Op::preset){preset_[c.limb]=c.value%3;return;}
  if(muted_[c.limb])return;
  auto& bank=voices_[c.limb];Voice* chosen=&bank[0];
  for(auto& v:bank){if(!v.active){chosen=&v;break;}if(v.age>chosen->age)chosen=&v;}
  *chosen={0,float(440*std::exp2((double(c.note)-69)/12)/rate),std::clamp(c.velocity,0.f,1.f),0,true,preset_[c.limb]};
 }
 void render(Shared& s,float* stereo,unsigned count,uint64_t dac_start_ns)noexcept{
  // A fixed ingestion budget bounds work even under an event flood.
  Command c;for(unsigned i=0;i<128 && s.commands.pop(c);++i){
   if(!scheduler_.add(c))s.scheduler_full.fetch_add(1,std::memory_order_relaxed);
  }
  for(unsigned i=0;i<4;++i){float cutoff=s.controls.cutoff[i].load(std::memory_order_relaxed);filters_[i].cutoff(std::isfinite(cutoff)?cutoff:8000);}
  float stem_gain=s.controls.stem_gain.load(std::memory_order_relaxed);
  stem_gain=std::isfinite(stem_gain)?std::clamp(stem_gain,0.f,1.f):0;
  bool alive=false;auto hb=s.controls.heartbeat_ns.load(std::memory_order_acquire);
  if(dac_start_ns>=hb)alive=hb && dac_start_ns-hb<250000000;
  if(!alive)for(auto& bank:voices_)for(auto& v:bank)v.active=false;
  const bool stems_active=s.stems_ready.load(std::memory_order_acquire)==1;
  for(unsigned n=0;n<count;++n,++frame_){
   while(scheduler_.due(frame_,c)){
    if(c.frame<frame_)s.late.fetch_add(1,std::memory_order_relaxed);
    if(alive){command(c);
     uint64_t audible=dac_start_ns+uint64_t(n)*1000000000/rate;
     if(c.capture_ns && audible>=c.capture_ns){auto latency=audible-c.capture_ns;auto max=s.max_capture_to_dac_ns.load(std::memory_order_relaxed);if(latency>max)s.max_capture_to_dac_ns.store(latency,std::memory_order_relaxed);}
    }
   }
   float sum=0;
   for(unsigned l=0;l<4;++l){float bus=0;
    for(auto& v:voices_[l])if(v.active){
     if(v.preset==2){if(v.age<drum_.size())bus+=drum_[v.age]*v.gain;else v.active=false;}
     else {unsigned idx=unsigned(v.phase*2048)&2047;float env=std::min(1.f,float(v.age)/96)*std::max(0.f,1.f-float(v.age)/24000);bus+=tables_[v.preset][idx]*v.gain*env*.18f;v.phase+=v.increment;v.phase-=std::floor(v.phase);}
     if(++v.age>=24000)v.active=false;
    }
    sum+=filters_[l].step(bus);
   }
   Stereo stem;bool got=stems_active && s.stem_audio.pop(stem);
   if(!got && stems_active)s.stem_starved.fetch_add(1,std::memory_order_relaxed);
   float left=sum+(got&&alive?stem.l*stem_gain:0),right=sum+(got&&alive?stem.r*stem_gain:0);
   // Zero-lookahead bounded saturation. This is not a true-peak brickwall limiter.
   stereo[2*n]=std::tanh(left*.7f);stereo[2*n+1]=std::tanh(right*.7f);
  }
 }
};
}
