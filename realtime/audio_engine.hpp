#pragma once
#include "dsp_primitives.hpp"
namespace cv_daw {
struct StemFrame {std::array<Stereo,4> stem{};};
struct StemBus {
 SpscQueue<StemFrame,8192> frames;
 std::atomic<float> requested_bpm{120};
 std::atomic<uint32_t> enabled{0};
};
struct RecordedHit {uint8_t note{};float velocity{};};
struct PatternSlot {std::array<RecordedHit,8> hits{};uint8_t count=0;};
class PendingCommands {
 struct Entry {AudioCommand command;uint64_t sequence;};
 std::array<Entry,256> heap_{};uint32_t count_=0;uint64_t sequence_=0;
 static bool earlier(const Entry& a,const Entry& b)noexcept{
  return a.command.target_frame<b.command.target_frame ||
   (a.command.target_frame==b.command.target_frame && a.sequence<b.sequence);
 }
public:
 bool push(AudioCommand command)noexcept{
  if(count_==heap_.size())return false;
  unsigned i=count_++;heap_[i]={command,sequence_++};
  while(i){unsigned parent=(i-1)/2;if(!earlier(heap_[i],heap_[parent]))break;std::swap(heap_[parent],heap_[i]);i=parent;}
  return true;
 }
 bool pop_due(uint64_t frame,AudioCommand& out)noexcept{
  if(!count_ || heap_[0].command.target_frame>frame)return false;
  out=heap_[0].command;heap_[0]=heap_[--count_];unsigned i=0;
  while(2*i+1<count_){
   unsigned child=2*i+1;
   if(child+1<count_ && earlier(heap_[child+1],heap_[child]))++child;
   if(!earlier(heap_[child],heap_[i]))break;
   std::swap(heap_[i],heap_[child]);i=child;
  }
  return true;
 }
};
class AudioEngine {
 SharedMemory& shared_;const Wavetables& tables_;std::array<SampleView,4> samples_;StemBus& stems_;
 std::array<VoicePool,4> voices_{};
 std::array<std::array<CytomicSvf,2>,4> filters_{};
 std::array<OnePoleSmoother,4> cutoff_{{OnePoleSmoother(8000),OnePoleSmoother(8000),OnePoleSmoother(8000),OnePoleSmoother(8000)}};
 std::array<OnePoleSmoother,4> send_{},bend_{};
 std::array<OnePoleSmoother,4> volume_{{OnePoleSmoother(.6f),OnePoleSmoother(.6f),OnePoleSmoother(.6f),OnePoleSmoother(.6f)}};
 OnePoleSmoother bpm_{120,50},energy_{.5f,20};
 Reverb reverb_;PendingCommands pending_;
 std::array<std::array<PatternSlot,32>,4> patterns_{};
 std::array<bool,4> muted_{},locked_{},stem_muted_{};
 std::array<uint8_t,4> preset_{{0,1,2,3}};
 uint64_t total_samples_rendered_=0,tick_index_=0,consumed_=0,invalid_=0,late_=0,overflow_=0,ticks_=0,starved_=0;
 double phase_=0;bool first_tick_=true;
 void trigger(unsigned limb,uint8_t note,float velocity)noexcept{
  if(muted_[limb])return;unsigned p=preset_[limb];SampleView asset=p>=2?samples_[p-2]:SampleView{};
  voices_[limb].note_on(note,velocity,p,tables_,asset);
 }
 void execute(const AudioCommand& c)noexcept{
  unsigned l=unsigned(c.limb_id);
  if(c.event_type==EventType::STEM_MUTE){if(l==4&&c.note_or_param<4)stem_muted_[c.note_or_param]=!stem_muted_[c.note_or_param];else ++invalid_;return;}
  if(l>=4){++invalid_;return;}
  switch(c.event_type){
  case EventType::NOTE_ON:{
   if(c.note_or_param>127 || c.value<0 || c.value>1){++invalid_;break;}
   if(!locked_[l]){
    trigger(l,c.note_or_param,c.value);
    if(!muted_[l]){unsigned slot=unsigned((tick_index_+(phase_>=.5?1:0))%32);auto& p=patterns_[l][slot];
     if(p.count<p.hits.size())p.hits[p.count++]={c.note_or_param,c.value};else {p.hits.back()={c.note_or_param,c.value};}
    }
   }break;
  }
  case EventType::NOTE_OFF:voices_[l].note_off(c.note_or_param);break;
  case EventType::MUTE_TOGGLE:muted_[l]=!muted_[l];if(muted_[l])voices_[l].silence();break;
  case EventType::LOOP_LOCK:locked_[l]=c.value>=.5f;break;
  case EventType::PRESET_CHANGE:if(c.note_or_param<6)preset_[l]=c.note_or_param;else ++invalid_;break;
  case EventType::STEM_MUTE:break;
  }
 }
public:
 AudioEngine(SharedMemory& s,const Wavetables& t,std::array<SampleView,4> samples,StemBus& stems):shared_(s),tables_(t),samples_(samples),stems_(stems){}
 uint64_t total_samples_rendered()const noexcept{return total_samples_rendered_;}
 double samples_per_sixteenth()const noexcept{return double(sample_rate)*15/bpm_.value();}
 // Pure bounded DSP/atomic memory operations. No clock calls, syscalls, allocators,
 // locks, library streaming/stretch calls, I/O, or waiting are reachable here.
 void processBlock(float* output,uint32_t frame_count)noexcept{
  if(!output || frame_count>block_frames)return;
  auto& controls=shared_.controls;
  std::array<float,4> fc{},rs{},pb{},sv{};
  for(unsigned l=0;l<4;++l){
   fc[l]=bounded(controls.limb_filter_cutoff[l].load(std::memory_order_relaxed),20,18000,8000);
   rs[l]=bounded(controls.limb_reverb_send[l].load(std::memory_order_relaxed),0,1,0);
   pb[l]=bounded(controls.limb_pitch_bend[l].load(std::memory_order_relaxed),-12,12,0);
   sv[l]=bounded(controls.stem_volumes[l].load(std::memory_order_relaxed),0,1,0);
  }
  float bpm=bounded(controls.target_bpm.load(std::memory_order_relaxed),60,200,120);
  float energy=bounded(controls.groove_energy.load(std::memory_order_relaxed),0,1,.5f);
  stems_.requested_bpm.store(bpm,std::memory_order_relaxed);
  AudioCommand command;
  for(unsigned i=0;i<command_capacity && shared_.commands.try_pop(command);++i){
   ++consumed_;if(!valid(command)){++invalid_;continue;}
   if(command.target_frame==0)command.target_frame=total_samples_rendered_;
   if(command.target_frame<total_samples_rendered_)++late_;
   // Reject absurd future timestamps rather than allowing one producer to reserve
   // the entire pending pool indefinitely. Ten seconds of scheduling is supported.
   if(command.target_frame>total_samples_rendered_+10*sample_rate){++invalid_;continue;}
   if(!pending_.push(command))++overflow_;
  }
  bool stems_enabled=stems_.enabled.load(std::memory_order_acquire)==1;
  for(unsigned n=0;n<frame_count;++n,++total_samples_rendered_){
   float current_bpm=bpm_.next(bpm),current_energy=energy_.next(energy);
   bool tick=first_tick_ || phase_>=1-1e-12;
   if(first_tick_)first_tick_=false;else if(tick){phase_=std::max(0.0,phase_-1);++tick_index_;}
   while(pending_.pop_due(total_samples_rendered_,command))execute(command);
   if(tick){++ticks_;for(unsigned l=0;l<4;++l)if(locked_[l]){
    auto& slot=patterns_[l][tick_index_%32];for(unsigned j=0;j<slot.count;++j)trigger(l,slot.hits[j].note,slot.hits[j].velocity);
   }}
   Stereo master{},reverb_in{};
   for(unsigned l=0;l<4;++l){
    float cutoff=cutoff_[l].next(fc[l]),send=send_[l].next(rs[l]),bend=tables_.bend(bend_[l].next(pb[l]));
    auto bus=voices_[l].next(tables_,bend);filters_[l][0].coefficients(cutoff);filters_[l][1].coefficients(cutoff);
    bus.l=filters_[l][0].next(bus.l).low;bus.r=filters_[l][1].next(bus.r).low;
    master.l+=bus.l;master.r+=bus.r;reverb_in.l+=bus.l*send;reverb_in.r+=bus.r*send;
   }
   auto wet=reverb_.next(reverb_in);master.l+=wet.l*.25f;master.r+=wet.r*.25f;
   StemFrame stems{};if(stems_enabled && !stems_.frames.try_pop(stems))++starved_;
   for(unsigned l=0;l<4;++l){float vol=volume_[l].next(stem_muted_[l]?0:sv[l]);master.l+=stems.stem[l].l*vol;master.r+=stems.stem[l].r*vol;}
   float gain=.4f+.2f*current_energy;output[2*n]=master.l*gain;output[2*n+1]=master.r*gain;
   phase_+=current_bpm/(sample_rate*15.0);
  }
  limit_stereo(output,frame_count);
  auto& t=shared_.telemetry;
  t.commands_consumed.store(consumed_,std::memory_order_relaxed);t.invalid_commands.store(invalid_,std::memory_order_relaxed);
  t.late_commands.store(late_,std::memory_order_relaxed);t.scheduler_overflows.store(overflow_,std::memory_order_relaxed);
  t.sixteenth_ticks.store(ticks_,std::memory_order_relaxed);t.stem_starved_frames.store(starved_,std::memory_order_relaxed);
  t.total_samples_rendered.store(total_samples_rendered_,std::memory_order_release);
 }
};
}
