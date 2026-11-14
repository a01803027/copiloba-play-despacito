#pragma once
#include "ipc.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
namespace cvdaw {
struct Quantized {uint64_t audible{},grid{};};
inline Quantized quantize(uint64_t event,uint64_t origin,double bpm,bool strict_latency=true){
 double tick=rate*15.0/std::clamp(bpm,60.0,200.0);
 double index=(double(event)-double(origin))/tick;
 double previous=double(origin)+std::floor(index)*tick, next=previous+tick;
 double late=double(event)-previous, early=next-double(event);
 uint64_t grid=static_cast<uint64_t>(std::max(0.0,late<=rate*.035?previous:next));
 // Strict policy only delays up to 8ms: total budget still requires measured perception <27ms.
 double window=rate*(strict_latency?.008:.040);
 return {late<=rate*.035 || early>window?event:static_cast<uint64_t>(next),grid};
}
inline uint8_t pitch(double height){
 constexpr std::array<uint8_t,7> scale{60,63,65,67,70,72,75};
 auto i=unsigned(std::clamp(int((height+1.5)/3.0*7),0,6));return scale[i];
}
struct PatternNote {bool used{};uint8_t note{};float velocity{};};
class Brain {
 Shared& s_; uint64_t last_frame_=0;double bpm_=120,phase_=0;bool started_=false;
 std::array<Mode,4> modes_{};
 std::array<uint8_t,4> gestures_{},presets_{{0,0,2,2}};
 std::array<std::array<PatternNote,32>,4> pattern_{};
 std::array<int64_t,4> next_tick_{{-1,-1,-1,-1}};
 bool strict_=true;
 bool send(Command c){if(s_.commands.push(c))return true;s_.drops.fetch_add(1,std::memory_order_relaxed);return false;}
public:
 explicit Brain(Shared& s,bool strict=true):s_(s),strict_(strict){}
 void step(uint64_t frame,uint64_t capture_ns,const std::array<bool,4>& hit,const std::array<double,4>& height,const std::array<uint8_t,4>& gesture,double bpm){
  if(started_ && frame<last_frame_)return;
  if(started_)phase_+=double(frame-last_frame_)/(rate*15.0/bpm_);
  started_=true;last_frame_=frame;bpm_=std::clamp(bpm,60.0,200.0);
  double tick=rate*15.0/bpm_;
  int64_t current=int64_t(std::floor(phase_));
  double previous_grid=double(frame)-(phase_-std::floor(phase_))*tick;
  uint64_t origin=uint64_t(std::max(0.0,previous_grid));
  for(unsigned i=0;i<4;++i){
   uint8_t g=gesture[i];
   if(g!=gestures_[i]){
    if(g==1){modes_[i]=Mode::muted;send({frame,0,Op::mute,uint8_t(i),0,1,0});}
    else if(g==2){modes_[i]=Mode::looping;next_tick_[i]=current+1;send({frame,0,Op::mute,uint8_t(i),0,0,0});}
    else if(g==3){presets_[i]=(presets_[i]+1)%3;send({frame,0,Op::preset,uint8_t(i),0,presets_[i],0});}
    else if(g==0){modes_[i]=Mode::live;send({frame,0,Op::mute,uint8_t(i),0,0,0});}
    gestures_[i]=g;
   }
   if(modes_[i]==Mode::live && hit[i]){
    auto q=quantize(frame,origin,bpm_,strict_);uint8_t n=pitch(height[i]);
    if(send({q.audible,capture_ns,Op::note,uint8_t(i),n,0,.8f})){
     int64_t idx=current+std::llround((double(q.grid)-previous_grid)/tick);
     pattern_[i][(idx%32+32)%32]={true,n,.8f};
    }
   }
   if(modes_[i]==Mode::looping){
    // Called at 60Hz. Schedule through the next vision interval, including sample offsets.
    int64_t end=current+int64_t(std::floor((double(frame+rate/50)-previous_grid)/tick));
    if(next_tick_[i]<current)next_tick_[i]=current;
    for(int64_t n=next_tick_[i];n<=end;++n){
     auto p=pattern_[i][(n%32+32)%32];
     if(p.used)send({uint64_t(std::max(0.0,previous_grid+(n-current)*tick)),0,Op::note,uint8_t(i),p.note,0,p.velocity});
    }
    next_tick_[i]=end+1;
   }
  }
  last_frame_=frame;
 }
};
}
