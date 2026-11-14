#pragma once
#include "ipc_bridge.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace cv_daw {
inline constexpr uint32_t sample_rate=48000,block_frames=128,voices_per_limb=8;
inline float bounded(float value,float lo,float hi,float fallback)noexcept {
 return finite(value)?std::clamp(value,lo,hi):fallback;
}
struct Stereo {float l{},r{};Stereo operator+(Stereo b)const noexcept{return {l+b.l,r+b.r};}};
class OnePoleSmoother {
 float state_,alpha_;
public:
 OnePoleSmoother():OnePoleSmoother(0,10){}
 explicit OnePoleSmoother(float initial,float milliseconds=10):state_(initial),alpha_(1.f-std::exp(-1.f/(sample_rate*milliseconds*.001f))){}
 float next(float target)noexcept{float updated=state_+alpha_*(target-state_);state_=updated==state_?target:updated;return state_;}
 float value()const noexcept{return state_;}
};
class Adsr {
 enum class Stage:uint8_t{off,attack,decay,sustain,release};Stage stage_=Stage::off;
 float level_=0,release_step_=0;
 static constexpr float attack_step=1.f/144,decay_step=(1.f-.55f)/3840,release_samples=5760;
public:
 void on()noexcept{stage_=Stage::attack;level_=0;}
 void off()noexcept{if(stage_!=Stage::off && stage_!=Stage::release){stage_=Stage::release;release_step_=level_/release_samples;}}
 bool active()const noexcept{return stage_!=Stage::off;}
 float next()noexcept{
  switch(stage_){
  case Stage::attack:level_+=attack_step;if(level_>=1){level_=1;stage_=Stage::decay;}break;
  case Stage::decay:level_-=decay_step;if(level_<=.55f){level_=.55f;stage_=Stage::sustain;}break;
  case Stage::sustain:break;
  case Stage::release:level_-=release_step_;if(level_<=1e-8f){level_=0;stage_=Stage::off;}break;
  case Stage::off:break;
  }return level_;
 }
};
struct SampleView {const float* samples=nullptr;uint32_t frames=0,channels=1;};
struct Wavetables {
 static constexpr uint32_t size=2048,bands=10;
 alignas(16) std::array<std::array<float,size+1>,bands> saw{};
 alignas(16) std::array<float,size+1> sine{};
 std::array<float,128> increments{};std::array<float,1025> bend_ratios{};
 Wavetables(){
  for(uint32_t i=0;i<size;++i){double phase=2*std::numbers::pi*i/size;sine[i]=float(std::sin(phase));
   for(uint32_t b=0;b<bands;++b){double sum=0;uint32_t harmonics=512u>>b;
    for(uint32_t h=1;h<=harmonics;++h)sum-=std::sin(phase*h)/h;
    saw[b][i]=float(sum*2/std::numbers::pi);
   }
  }
  sine[size]=sine[0];for(auto& b:saw)b[size]=b[0];
  for(unsigned n=0;n<128;++n)increments[n]=float(440*std::exp2((double(n)-69)/12)/sample_rate);
  for(unsigned i=0;i<bend_ratios.size();++i)bend_ratios[i]=float(std::exp2((-12.+24.*i/1024)/12));
 }
 float bend(float semitones)const noexcept{
  float x=(semitones+12)*(1024.f/24);unsigned i=std::min(1023u,unsigned(x));return bend_ratios[i]+(bend_ratios[i+1]-bend_ratios[i])*(x-i);
 }
 float oscillator(float phase,unsigned band,bool is_sine)const noexcept{
  float x=phase*size;unsigned i=unsigned(x);float frac=x-i;
  const float* p=is_sine?sine.data():saw[band].data();return p[i]+(p[i+1]-p[i])*frac;
 }
 unsigned band_for(float increment)const noexcept{
  // Reserve an octave of headroom for the full +12-semitone bend range.
  unsigned b=0;while(b+1<bands && (512u>>b)*increment*2>.45f)++b;return b;
 }
};
// TPT/Cytomic SVF. Positive g and k preserve topology stability for every supported cutoff.
class CytomicSvf {
 float ic1_=0,ic2_=0,a1_=1,a2_=0,a3_=0,k_=1.41421356f;
public:
 void coefficients(float cutoff,float resonance=.70710678f)noexcept{
  float x=float(std::numbers::pi)*cutoff/sample_rate,x2=x*x;
  // [5/4] rational tan approximation, x in [0.0013,1.1781].
  float g=x*(105-10*x2)/(105-45*x2+x2*x2);k_=1/std::clamp(resonance,.5f,4.f);
  a1_=1/(1+g*(g+k_));a2_=g*a1_;a3_=g*a2_;
 }
 struct Outputs{float low,band,high;};
 Outputs next(float in)noexcept{
  float v3=in-ic2_,v1=a1_*ic1_+a2_*v3,v2=ic2_+a2_*ic1_+a3_*v3;
  ic1_=2*v1-ic1_;ic2_=2*v2-ic2_;return {v2,v1,in-k_*v1-v2};
 }
};
struct Voice {
 Adsr envelope;SampleView sample{};float phase=0,increment=0,gain=0;double sample_position=0;
 uint32_t age=0;uint8_t note=0,band=0;bool active=false,is_sample=false,is_sine=false;
 void start(uint8_t n,float velocity,unsigned preset,const Wavetables& tables,SampleView asset)noexcept{
  note=n;phase=0;sample_position=0;age=0;increment=tables.increments[n];band=uint8_t(tables.band_for(increment));
  gain=velocity;is_sample=preset>=2;is_sine=preset==1;sample=asset;active=!is_sample || (sample.samples && sample.frames);envelope.on();
 }
 void off()noexcept{envelope.off();}
 Stereo next(const Wavetables& tables,float bend_ratio)noexcept{
  if(!active)return {};
  float env=envelope.next();Stereo out{};
  if(is_sample){
   uint32_t i=uint32_t(sample_position);
   if(i>=sample.frames){active=false;return {};}
   uint32_t j=std::min(i+1,sample.frames-1);float frac=float(sample_position-i);
   auto read=[&](unsigned channel){unsigned c=sample.channels==1?0:channel;float a=sample.samples[i*sample.channels+c],b=sample.samples[j*sample.channels+c];return a+(b-a)*frac;};
   out={read(0)*gain*env,read(1)*gain*env};sample_position+=bend_ratio;
   if(!envelope.active())active=false;
  }else{
   float value=tables.oscillator(phase,band,is_sine)*env*gain*.18f;out={value,value};
   phase+=std::min(increment*bend_ratio,.45f);if(phase>=1)phase-=1;
  }
  ++age;if(age==2*sample_rate)envelope.off();if(!envelope.active())active=false;return out;
 }
};
class VoicePool {
 std::array<Voice,voices_per_limb> voices_{};
public:
 void note_on(uint8_t n,float value,unsigned preset,const Wavetables& t,SampleView sample)noexcept{
  Voice* selected=&voices_[0];for(auto& v:voices_){if(!v.active){selected=&v;break;}if(v.age>selected->age)selected=&v;}
  selected->start(n,value,preset,t,sample);
 }
 void note_off(uint8_t n)noexcept{for(auto& v:voices_)if(v.active&&v.note==n)v.off();}
 void silence()noexcept{for(auto& v:voices_)v.active=false;}
 Stereo next(const Wavetables& tables,float bend_ratio)noexcept{
  Stereo sum{};for(auto& v:voices_){auto x=v.next(tables,bend_ratio);sum.l+=x.l;sum.r+=x.r;}return sum;
 }
};
class Reverb {
 std::array<float,1499> left_{};std::array<float,2111> right_{};unsigned l_=0,r_=0;
public:
 Stereo next(Stereo input)noexcept{
  Stereo delayed{left_[l_],right_[r_]};left_[l_]=input.l+.68f*delayed.r;right_[r_]=input.r+.68f*delayed.l;
  if(++l_==left_.size())l_=0;if(++r_==right_.size())r_=0;return delayed;
 }
};
inline float soft_clip(float x)noexcept{
 if(!finite(x))return 0;x=std::clamp(x,-3.f,3.f);float square=x*x;
 return std::clamp(x*(27+square)/(27+9*square),-1.f,1.f);
}
inline void limit_stereo(float* samples,uint32_t frames)noexcept{
 unsigned n=0;
#if defined(__aarch64__)
 // Four independent samples = two stereo frames per NEON iteration.
 const auto lo=vdupq_n_f32(-3),hi=vdupq_n_f32(3),c27=vdupq_n_f32(27),c9=vdupq_n_f32(9);
 for(;n+4<=frames*2;n+=4){
  auto x=vld1q_f32(samples+n);auto bits=vreinterpretq_u32_f32(x);
  auto normal=vcltq_u32(vandq_u32(bits,vdupq_n_u32(0x7f800000)),vdupq_n_u32(0x7f800000));
  x=vbslq_f32(normal,x,vdupq_n_f32(0));x=vminq_f32(hi,vmaxq_f32(lo,x));auto square=vmulq_f32(x,x);
  auto y=vdivq_f32(vmulq_f32(x,vaddq_f32(c27,square)),vaddq_f32(c27,vmulq_f32(c9,square)));
  vst1q_f32(samples+n,vminq_f32(vdupq_n_f32(1),vmaxq_f32(vdupq_n_f32(-1),y)));
 }
#endif
 for(;n<frames*2;++n)samples[n]=soft_clip(samples[n]);
}
}
