#pragma once
#include "audio_engine.hpp"
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
#ifdef CV_DAW_WITH_RUBBERBAND
#include <rubberband/RubberBandStretcher.h>
#endif
namespace cv_daw {
// All four decoded arrays are supplied at startup and remain immutable.
// Ratio = source_BPM / target_BPM (duration multiplier, not speed multiplier).
class StemWorker {
 StemBus& bus_;std::array<std::vector<float>,4> source_;
 std::atomic<bool> stop_{false},failed_{false};std::thread worker_;[[maybe_unused]] double source_bpm_;
 bool fixed_tempo_;
#ifdef CV_DAW_WITH_RUBBERBAND
 using Stretcher=RubberBand::RubberBandStretcher;
 struct Stream {
  std::unique_ptr<Stretcher> stretch;
  std::array<float,1024> in_l{},in_r{},out_l{},out_r{};
  size_t cursor=0,skip=0;
 };
 std::array<Stream,4> streams_;
 void trim(Stream& s){
  while(s.skip && s.stretch->available()>0){size_t n=std::min({s.skip,size_t(s.stretch->available()),size_t(1024)});
   float* out[]{s.out_l.data(),s.out_r.data()};auto got=s.stretch->retrieve(out,n);s.skip-=got;if(!got)break;
  }
 }
 void feed(unsigned index){auto& s=streams_[index];auto& data=source_[index];
  for(unsigned i=0;i<1024;++i){s.in_l[i]=data[2*s.cursor];s.in_r[i]=data[2*s.cursor+1];s.cursor=(s.cursor+1)%(data.size()/2);}
  const float* in[]{s.in_l.data(),s.in_r.data()};s.stretch->process(in,1024,false);trim(s);
 }
#endif
 void run()noexcept{
#ifdef __linux__
  cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(0,&cpus);
  if(!fixed_tempo_ && pthread_setaffinity_np(pthread_self(),sizeof(cpus),&cpus)){failed_.store(true);return;}
#endif
  try{
   size_t cursor=0;
#ifdef CV_DAW_WITH_RUBBERBAND
   double previous=bus_.requested_bpm.load(std::memory_order_relaxed);
#endif
   while(!stop_.load(std::memory_order_relaxed)){
    auto queued=bus_.frames.size_for_producer();
    if(queued>=4096){bus_.enabled.store(1,std::memory_order_release);std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
    if(fixed_tempo_){
     const size_t length=source_[0].size()/2;
     for(unsigned i=0;i<1024;++i){
      StemFrame frame;
      for(unsigned j=0;j<4;++j)frame.stem[j]={source_[j][2*cursor],source_[j][2*cursor+1]};
      if(!bus_.frames.try_push(frame))throw std::runtime_error("stem FIFO overflow");
      cursor=(cursor+1)%length;
     }
     continue;
    }
#ifdef CV_DAW_WITH_RUBBERBAND
    double bpm=bounded(bus_.requested_bpm.load(std::memory_order_relaxed),60,200,120);
    // Limit modulation slew to prevent sudden stretch artifacts. This work and
    // any Rubber Band internal allocation never execute in the audio callback.
    double ratio_bpm=std::clamp(bpm,previous-.5,previous+.5);previous=ratio_bpm;
    size_t count=1024;
    for(unsigned i=0;i<4;++i){auto& s=streams_[i];s.stretch->setTimeRatio(source_bpm_/ratio_bpm);trim(s);
     unsigned iterations=0;while(s.stretch->available()<1024 && iterations++<32)feed(i);
     count=std::min(count,size_t(std::max(0,s.stretch->available())));
    }
    if(!count){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
    for(auto& s:streams_){float* out[]{s.out_l.data(),s.out_r.data()};if(s.stretch->retrieve(out,count)!=count)throw std::runtime_error("short stretch retrieve");}
    for(size_t i=0;i<count;++i){StemFrame frame;for(unsigned j=0;j<4;++j)frame.stem[j]={streams_[j].out_l[i],streams_[j].out_r[i]};
     if(!bus_.frames.try_push(frame))throw std::runtime_error("stem FIFO overflow");
    }
#endif
   }
  }catch(...){failed_.store(true,std::memory_order_release);}
 }
public:
 StemWorker(StemBus& bus,std::array<std::vector<float>,4> sources,double source_bpm,double initial_bpm=120,bool fixed_tempo=false):bus_(bus),source_(std::move(sources)),source_bpm_(source_bpm),fixed_tempo_(fixed_tempo){
  if(fixed_tempo_ && source_bpm!=initial_bpm)throw std::runtime_error("fixed stems require matching BPM");
  if((std::bit_cast<uint64_t>(source_bpm)&0x7ff0000000000000ull)==0x7ff0000000000000ull || source_bpm<60 || source_bpm>200)throw std::runtime_error("stem source BPM must be 60..200");
  for(auto& data:source_)if(data.empty() || data.size()%2 || data.size()!=source_[0].size())throw std::runtime_error("four equal-length stereo stems required");
  if(!fixed_tempo_){
#ifdef CV_DAW_WITH_RUBBERBAND
  for(auto& s:streams_){
   s.stretch=std::make_unique<Stretcher>(sample_rate,2,Stretcher::OptionProcessRealTime|Stretcher::OptionThreadingNever|Stretcher::OptionEngineFaster|Stretcher::OptionWindowShort,source_bpm/initial_bpm);
   s.stretch->setMaxProcessSize(1024);s.skip=s.stretch->getStartDelay();size_t pad=s.stretch->getPreferredStartPad();const float* zeros[]{s.in_l.data(),s.in_r.data()};
   while(pad){size_t count=std::min(pad,size_t(1024));s.stretch->process(zeros,count,false);pad-=count;}
  }
#else
  throw std::runtime_error("rebuild with CV_DAW_WITH_RUBBERBAND=ON to enable time stretching");
#endif
  }
  worker_=std::thread([this]{run();});
 }
 ~StemWorker(){stop_.store(true);if(worker_.joinable())worker_.join();}
 bool failed()const noexcept{return failed_.load(std::memory_order_acquire);}
 void wait_for_prebuffer(){
  for(unsigned i=0;i<5000;++i){if(failed())throw std::runtime_error("stem worker failed");if(bus_.enabled.load(std::memory_order_acquire))return;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
  throw std::runtime_error("stem prebuffer timeout");
 }
};
}
