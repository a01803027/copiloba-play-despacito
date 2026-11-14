#include "cvdaw/motion.hpp"
#include "cvdaw/music.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <sched.h>
#include <memory>
#include <thread>
#include <chrono>
using namespace cvdaw;
static uint64_t now(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
int main(int argc,char** argv){try{
#ifdef __linux__
 cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(0,&cpus);if(sched_setaffinity(0,sizeof(cpus),&cpus))throw std::runtime_error("brain affinity");
#endif
 std::unique_ptr<Mapping> mapping;
 for(unsigned attempt=0;attempt<100;++attempt){
  try{mapping=std::make_unique<Mapping>("/cvdaw",false);}catch(const std::runtime_error&){ }
  if(mapping && mapping->get().audio_ready.load(std::memory_order_acquire))break;
  mapping.reset();std::this_thread::sleep_for(std::chrono::milliseconds(50));
 }
 if(!mapping)throw std::runtime_error("audio not ready after 5 seconds");
 auto& s=mapping->get();
 if(s.brain_ready.exchange(1))throw std::runtime_error("a brain is already connected");
 Brain brain(s,!(argc>1 && std::string(argv[1])=="--full-quantize"));Motion motion;Temporal temporal;
 std::string line;
 // Protocol: capture_monotonic_ns, 99 aspect-corrected pose floats, four posture integers,
 // optional energy BPM and groove probabilities from the slow model worker.
 while(std::getline(std::cin,line)){
  std::istringstream in(line);uint64_t ns;Pose pose{};std::array<uint8_t,4> gestures{};bool ok=bool(in>>ns);
  for(auto& p:pose){ok=ok&&bool(in>>p.x>>p.y>>p.z);ok=ok&&std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}
  for(auto& g:gestures){int v=-1;ok=ok&&bool(in>>v)&&v>=0&&v<=3;g=uint8_t(v);}
  uint64_t current=now();if(!ok||ns>current||current-ns>100000000){motion.reset();continue;}
  MotionFrame f;if(!motion.process(pose,ns,f))continue;
  temporal.step(f,double(ns)*1e-9);float energy=temporal.energy,bpm=temporal.bpm;
  float model_energy;
  if(in>>model_energy){
   if(std::isfinite(model_energy))energy=std::clamp(model_energy,0.f,1.f);
   for(auto& probability:s.controls.groove){float value=0;if(in>>value && std::isfinite(value))probability.store(std::clamp(value,0.f,1.f),std::memory_order_relaxed);}
  }
  s.controls.energy.store(energy,std::memory_order_relaxed);s.controls.bpm.store(bpm,std::memory_order_relaxed);
  auto clock=clock_snapshot(s);if(!clock.ns)continue;
  // ALSA publishes frame-to-DAC time including queued hardware delay. Earliest writable
  // frame is clock.frame; past capture timestamps cannot be scheduled retroactively.
  uint64_t frame=clock.frame;
  if(current>clock.ns)frame+=uint64_t(double(current-clock.ns)*rate/1e9);
  std::array<double,4> height{};
  for(unsigned i=0;i<4;++i){height[i]=-f.position[i].y;s.controls.cutoff[i].store(float(200+std::clamp(f.radius[i]/3.,0.,1.)*12000),std::memory_order_relaxed);}
  s.controls.heartbeat_ns.store(current,std::memory_order_release);
  brain.step(frame,ns,f.hit,height,gestures,bpm);
 }
 s.controls.heartbeat_ns.store(0,std::memory_order_release);s.brain_ready.store(0,std::memory_order_release);
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
