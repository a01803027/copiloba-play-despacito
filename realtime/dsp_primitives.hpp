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
inline Stereo apply_pan(Stereo in,float pan)noexcept{
 float p=std::clamp(pan,-1.f,1.f);
 float left_gain=std::sqrt(0.5f*(1.f-p));
 float right_gain=std::sqrt(0.5f*(1.f+p));
 // Preserve mono center loudness (~1.0 when pan==0).
 constexpr float norm=1.41421356f;
 return {in.l*left_gain*norm,in.r*right_gain*norm};
}
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
 float attack_step_=1.f/144.f,decay_step_=(1.f-.55f)/3840.f,sustain_=.55f,release_samples_=5760.f;
public:
 void configure(float attack_ms,float decay_ms,float sustain,float release_ms)noexcept{
  float a_samp=std::max(1.f,sample_rate*attack_ms*.001f);
  float d_samp=std::max(1.f,sample_rate*decay_ms*.001f);
  sustain_=std::clamp(sustain,0.f,1.f);
  attack_step_=1.f/a_samp;
  decay_step_=(1.f-sustain_)/d_samp;
  release_samples_=std::max(1.f,sample_rate*release_ms*.001f);
 }
 void on()noexcept{stage_=Stage::attack;level_=0;}
 void off()noexcept{if(stage_!=Stage::off && stage_!=Stage::release){stage_=Stage::release;release_step_=level_/release_samples_;}}
 bool active()const noexcept{return stage_!=Stage::off;}
 float next()noexcept{
  switch(stage_){
  case Stage::attack:level_+=attack_step_;if(level_>=1){level_=1;stage_=Stage::decay;}break;
  case Stage::decay:level_-=decay_step_;if(level_<=sustain_){level_=sustain_;stage_=Stage::sustain;}break;
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
 alignas(16) std::array<std::array<float,size+1>,bands> tri{};
 alignas(16) std::array<float,size+1> sine{};
 std::array<float,128> increments{};std::array<float,1025> bend_ratios{};
 Wavetables(){
  for(uint32_t i=0;i<size;++i){double phase=2*std::numbers::pi*i/size;sine[i]=float(std::sin(phase));
   for(uint32_t b=0;b<bands;++b){double sum_saw=0,sum_tri=0;uint32_t harmonics=512u>>b;
    for(uint32_t h=1;h<=harmonics;++h){
     sum_saw-=std::sin(phase*h)/h;
     if(h&1){double sign=((h>>1)&1)?-1.0:1.0;sum_tri+=sign*std::sin(phase*h)/(double(h)*double(h));}
    }
    saw[b][i]=float(sum_saw*2/std::numbers::pi);
    tri[b][i]=float(sum_tri*8/(std::numbers::pi*std::numbers::pi));
   }
  }
  sine[size]=sine[0];for(auto& b:saw)b[size]=b[0];for(auto& b:tri)b[size]=b[0];
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
 float triangle(float phase,unsigned band)const noexcept{
  float x=phase*size;unsigned i=unsigned(x);float frac=x-i;
  const float* p=tri[band].data();return p[i]+(p[i+1]-p[i])*frac;
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
 Adsr envelope;SampleView sample{};
 float phase=0,phase2=0,phase3=0,increment=0,gain=0,mod_env=0;
 double sample_position=0;
 uint32_t age=0,max_age=2*sample_rate,noise=1;
 uint8_t note=0,band=0,preset_id=0,drum=0;
 bool active=false,is_sample=false,is_sine=false;
 void start(uint8_t n,float velocity,unsigned preset,const Wavetables& tables,SampleView asset)noexcept{
  note=n;phase=0;phase2=0.17f;phase3=0.61f;sample_position=0;age=0;
  increment=tables.increments[n];band=uint8_t(tables.band_for(increment));
  gain=velocity;preset_id=uint8_t(preset);
  drum=(preset==6 || preset==7)?uint8_t(preset-5):0;
  noise=1;is_sample=preset>=2 && preset<6;is_sine=preset==1;
  sample=asset;active=!is_sample || (sample.samples && sample.frames);
  mod_env=1.0f;
  // Configure musical envelopes per preset while keeping presets 0..7 unchanged.
  if(preset==8){ // Warm Unison Pad
   envelope.configure(160.f,450.f,.78f,850.f);max_age=uint32_t(4.5f*sample_rate);
  }else if(preset==9){ // FM Warm Keys (Electric Piano)
   envelope.configure(4.f,650.f,.32f,360.f);max_age=uint32_t(3.5f*sample_rate);
  }else if(preset==10){ // Supersaw / Pluck Lead
   envelope.configure(3.f,220.f,.28f,220.f);max_age=2*sample_rate;
  }else if(preset==11){ // Deep Sub / Warm Bass
   envelope.configure(5.f,180.f,.68f,140.f);max_age=2*sample_rate;
  }else{
   envelope.configure(3.f,80.f,.55f,120.f);max_age=2*sample_rate;
  }
  envelope.on();
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
  }else if(drum){
   constexpr unsigned duration=sample_rate/5;
   float tail=1-float(age)/duration;
   noise^=noise<<13;noise^=noise>>17;noise^=noise<<5;
   float tone=tables.oscillator(phase,band,true);
   float white=float(noise>>8)*(2.f/16777216.f)-1;
   float value=(drum==1?tone:.75f*white+.25f*tone)*tail*tail*gain*.55f;
   out={value,value};
   phase+=increment*bend_ratio*(1+tail*tail*(drum==1?3:1));phase-=unsigned(phase);
   if(age+1==duration)active=false;
  }else if(preset_id==8){
   // Warm Unison Pad: 3 detuned stereo oscillators (saw + triangle warmth).
   float inc=std::min(increment*bend_ratio,.45f);
   float s_c=0.55f*tables.triangle(phase,band)+0.45f*tables.oscillator(phase,band,false);
   float s_l=0.60f*tables.oscillator(phase2,band,false)+0.40f*tables.triangle(phase2,band);
   float s_r=0.60f*tables.oscillator(phase3,band,false)+0.40f*tables.triangle(phase3,band);
   float scale=env*gain*0.14f;
   out={(s_c*0.55f+s_l*0.75f+s_r*0.25f)*scale,(s_c*0.55f+s_r*0.75f+s_l*0.25f)*scale};
   phase+=inc;if(phase>=1.f)phase-=1.f;
   phase2+=std::min(inc*0.9948f,.45f);if(phase2>=1.f)phase2-=1.f; // -9 cents
   phase3+=std::min(inc*1.0052f,.45f);if(phase3>=1.f)phase3-=1.f; // +9 cents
  }else if(preset_id==9){
   // FM Warm Keys (Rhodes-like 2-op FM bell attack + warm body).
   float inc=std::min(increment*bend_ratio,.45f);
   mod_env*=0.99985f;
   float mod=tables.oscillator(phase2,band,true)*(0.12f+1.35f*mod_env*mod_env);
   float p_mod=phase+mod*0.16f;p_mod-=std::floor(p_mod);
   float body=tables.oscillator(p_mod,band,true);
   float shimmer=tables.oscillator(phase3,band,true)*0.12f*mod_env;
   float scale=env*gain*0.22f;
   out={(body+shimmer)*scale,(body-shimmer*0.7f)*scale};
   phase+=inc;if(phase>=1.f)phase-=1.f;
   phase2+=std::min(inc*2.0f,.45f);if(phase2>=1.f)phase2-=1.f;
   phase3+=std::min(inc*3.004f,.45f);if(phase3>=1.f)phase3-=1.f;
  }else if(preset_id==10){
   // Supersaw / Pluck Lead: stereo detuned saws + warm fundamental sine.
   float inc=std::min(increment*bend_ratio,.45f);
   mod_env*=0.99975f;
   float sl=tables.oscillator(phase2,band,false);
   float sr=tables.oscillator(phase3,band,false);
   float sub=tables.oscillator(phase,band,true);
   float bright=0.35f+0.65f*mod_env;
   float scale=env*gain*0.16f;
   out={(sub*0.45f+(sl*0.75f+sr*0.25f)*bright)*scale,(sub*0.45f+(sr*0.75f+sl*0.25f)*bright)*scale};
   phase+=inc;if(phase>=1.f)phase-=1.f;
   phase2+=std::min(inc*0.9960f,.45f);if(phase2>=1.f)phase2-=1.f; // -7 cents
   phase3+=std::min(inc*1.0040f,.45f);if(phase3>=1.f)phase3-=1.f; // +7 cents
  }else if(preset_id==11){
   // Deep Sub / Warm Bass: clean sine sub + subtle warm harmonic.
   float inc=std::min(increment*bend_ratio,.45f);
   float sub=tables.oscillator(phase,band,true);
   float harm=tables.triangle(phase2,band)*0.30f;
   float v=(sub+harm)*env*gain*0.26f;
   out={v,v};
   phase+=inc;if(phase>=1.f)phase-=1.f;
   phase2+=std::min(inc*2.0f,.45f);if(phase2>=1.f)phase2-=1.f;
  }else{
   float value=tables.oscillator(phase,band,is_sine)*env*gain*.18f;out={value,value};
   phase+=std::min(increment*bend_ratio,.45f);if(phase>=1)phase-=1;
  }
  ++age;if(age==max_age)envelope.off();if(!envelope.active())active=false;return out;
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
// 4-channel Hadamard FDN Stereo Reverb with 1-pole low-pass damping (zero heap allocs).
class Reverb {
 std::array<float,2069> d0_{};
 std::array<float,2741> d1_{};
 std::array<float,3511> d2_{};
 std::array<float,4373> d3_{};
 std::array<float,4> lp_{};
 unsigned i0_=0,i1_=0,i2_=0,i3_=0;
public:
 Stereo next(Stereo input)noexcept{
  float y0=d0_[i0_],y1=d1_[i1_],y2=d2_[i2_],y3=d3_[i3_];
  // 1-pole lowpass damping inside feedback loop for warm ambient tail.
  constexpr float damp=0.38f,fb=0.78f;
  lp_[0]+=damp*(y0-lp_[0]);lp_[1]+=damp*(y1-lp_[1]);
  lp_[2]+=damp*(y2-lp_[2]);lp_[3]+=damp*(y3-lp_[3]);
  float a=lp_[0],b=lp_[1],c=lp_[2],d=lp_[3];
  // Normalized 4x4 Hadamard orthogonal mixing matrix (factor 0.5).
  float f0=0.5f*(a+b+c+d)*fb;
  float f1=0.5f*(a-b+c-d)*fb;
  float f2=0.5f*(a+b-c-d)*fb;
  float f3=0.5f*(a-b-c+d)*fb;
  d0_[i0_]=input.l+f0;d1_[i1_]=input.r+f1;
  d2_[i2_]=input.l*0.7f+input.r*0.3f+f2;
  d3_[i3_]=input.r*0.7f+input.l*0.3f+f3;
  if(++i0_==d0_.size())i0_=0;if(++i1_==d1_.size())i1_=0;
  if(++i2_==d2_.size())i2_=0;if(++i3_==d3_.size())i3_=0;
  return {0.5f*(y0+y2),0.5f*(y1+y3)};
 }
};
// Tempo-synced Stereo Ping-Pong Delay (fixed 1-second stereo buffer, zero heap allocs).
class StereoDelay {
 static constexpr uint32_t max_delay=sample_rate;
 std::array<float,max_delay> left_{},right_{};
 uint32_t write_=0;
 float lp_l_=0,lp_r_=0;
public:
 Stereo next(Stereo input,float bpm)noexcept{
  // Dotted-eighth note delay (3/16 of a bar = 0.75 * 60 / bpm seconds).
  float seconds=std::clamp(45.f/std::clamp(bpm,60.f,200.f),0.08f,0.95f);
  uint32_t delay_samples=std::clamp(uint32_t(seconds*sample_rate),1u,max_delay-1);
  uint32_t read=(write_+max_delay-delay_samples)%max_delay;
  float dl=left_[read],dr=right_[read];
  constexpr float damp=0.45f,fb=0.42f;
  lp_l_+=damp*(dl-lp_l_);lp_r_+=damp*(dr-lp_r_);
  left_[write_]=input.l+lp_r_*fb;
  right_[write_]=input.r+lp_l_*fb;
  if(++write_==max_delay)write_=0;
  return {dl,dr};
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
