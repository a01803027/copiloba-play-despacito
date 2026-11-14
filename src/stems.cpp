#include "cvdaw/ipc.hpp"
#include <rubberband/RubberBandStretcher.h>
#include <sndfile.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <sched.h>
using namespace cvdaw;
static volatile sig_atomic_t running=1;
static void stop(int){running=0;}
using RB=RubberBand::RubberBandStretcher;
struct Stem {
 std::vector<float> l,r;size_t cursor=0;std::unique_ptr<RB> rb;
 std::array<float,1024> in_l{},in_r{},out_l{},out_r{};
 size_t skip=0;
 explicit Stem(const char* path,double ratio){
  SF_INFO info{};SNDFILE* f=sf_open(path,SFM_READ,&info);
  if(!f)throw std::runtime_error("cannot open stem");
  if(info.samplerate!=int(rate)||info.channels!=2||info.frames<=0||info.frames>int64_t(rate)*60*15){sf_close(f);throw std::runtime_error("stems must be 48k stereo, <=15 minutes");}
  std::vector<float> inter(size_t(info.frames)*2);auto got=sf_readf_float(f,inter.data(),info.frames);sf_close(f);
  if(got!=info.frames)throw std::runtime_error("short stem read");
  l.resize(got);r.resize(got);for(size_t i=0;i<l.size();++i){l[i]=inter[2*i];r[i]=inter[2*i+1];if(!std::isfinite(l[i])||!std::isfinite(r[i]))throw std::runtime_error("nonfinite stem");}
  rb=std::make_unique<RB>(rate,2,RB::OptionProcessRealTime|RB::OptionThreadingNever|RB::OptionEngineFaster|RB::OptionWindowShort,ratio);
  rb->setMaxProcessSize(1024);
  skip=rb->getStartDelay();
  size_t pad=rb->getPreferredStartPad();const float* zero[]{in_l.data(),in_r.data()};
  while(pad){size_t n=std::min(pad,size_t(1024));rb->process(zero,n,false);pad-=n;}
 }
 void trim(){
  while(skip && rb->available()>0){size_t n=std::min({skip,size_t(rb->available()),size_t(1024)});float* out[]{out_l.data(),out_r.data()};size_t got=rb->retrieve(out,n);skip-=got;if(!got)break;}
 }
 void feed(){for(unsigned i=0;i<1024;++i){in_l[i]=l[cursor];in_r[i]=r[cursor];cursor=(cursor+1)%l.size();}const float* in[]{in_l.data(),in_r.data()};rb->process(in,1024,false);}
};
int main(int argc,char** argv){try{
 if(argc!=6)throw std::runtime_error("usage: cvdaw-stems SOURCE_BPM stem1.wav stem2.wav stem3.wav stem4.wav");
 double source=std::stod(argv[1]);if(!std::isfinite(source)||source<60||source>200)throw std::runtime_error("source BPM 60..200");
 cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(0,&cpus);if(sched_setaffinity(0,sizeof(cpus),&cpus))throw std::runtime_error("stem affinity");
 Mapping mapping("/cvdaw",false);auto& s=mapping.get();
 std::array<std::unique_ptr<Stem>,4> stems;for(unsigned i=0;i<4;++i)stems[i]=std::make_unique<Stem>(argv[i+2],source/120.0);
 for(unsigned i=1;i<4;++i)if(stems[i]->l.size()!=stems[0]->l.size())throw std::runtime_error("stems must have identical lengths and aligned downbeats");
 uint32_t expected=0;
 if(!s.stems_ready.compare_exchange_strong(expected,2,std::memory_order_acq_rel))throw std::runtime_error("stem producer already connected");
 signal(SIGINT,stop);signal(SIGTERM,stop);
 // Variable-cost Rubber Band runs on Core 0. Core 1 only consumes fixed-size frames.
 // Buffering decouples stretch latency from live notes, at a cost to tempo response.
 while(running && s.audio_ready.load(std::memory_order_acquire)){
  auto queued=s.stem_audio.write.load(std::memory_order_relaxed)-s.stem_audio.read.load(std::memory_order_acquire);
  if(queued>4096){std::this_thread::sleep_for(std::chrono::milliseconds(2));continue;}
  double bpm=s.controls.bpm.load(std::memory_order_relaxed);if(!std::isfinite(bpm))bpm=120;
  size_t available=1024;
  for(auto& p:stems){p->rb->setTimeRatio(source/std::clamp(bpm,60.,200.));unsigned feeds=0;p->trim();while(p->rb->available()<1024 && feeds++<32){p->feed();p->trim();}available=std::min(available,size_t(std::max(0,p->rb->available())));}
  if(!available)continue;
  for(auto& p:stems){float* out[]{p->out_l.data(),p->out_r.data()};if(p->rb->retrieve(out,available)!=available)throw std::runtime_error("Rubber Band short retrieve");}
  for(size_t n=0;n<available;++n){Stereo frame{};for(auto& p:stems){frame.l+=p->out_l[n]*.25f;frame.r+=p->out_r[n]*.25f;}
   if(!s.stem_audio.push(frame))throw std::runtime_error("stem FIFO full");}
  if(s.stems_ready.load()==2 && s.stem_audio.write.load()-s.stem_audio.read.load()>=4096)s.stems_ready.store(1,std::memory_order_release);
 }
 s.stems_ready.store(0,std::memory_order_release);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
