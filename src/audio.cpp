#include "cvdaw/dsp.hpp"
#include <alsa/asoundlib.h>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
using namespace cvdaw;
static volatile sig_atomic_t running=1;
static void stop(int){running=0;}
static uint64_t now(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
static void check(int r){if(r<0)throw std::runtime_error(snd_strerror(r));}
struct Pcm {snd_pcm_t* p=nullptr;~Pcm(){if(p)snd_pcm_close(p);}};
int main(int argc,char** argv){try{
 unsigned period=argc>1?unsigned(std::stoul(argv[1])):128;
 if(period!=128 && period!=256)throw std::runtime_error("period must be 128 or 256");
 Pcm pcm;check(snd_pcm_open(&pcm.p,"hw:0,0",SND_PCM_STREAM_PLAYBACK,0));
 snd_pcm_hw_params_t* hw;snd_pcm_hw_params_alloca(&hw);check(snd_pcm_hw_params_any(pcm.p,hw));
 check(snd_pcm_hw_params_set_access(pcm.p,hw,SND_PCM_ACCESS_RW_INTERLEAVED));
 snd_pcm_format_t format=SND_PCM_FORMAT_FLOAT_LE;
 if(snd_pcm_hw_params_test_format(pcm.p,hw,format)<0)format=SND_PCM_FORMAT_S32_LE;
 check(snd_pcm_hw_params_set_format(pcm.p,hw,format));
 check(snd_pcm_hw_params_set_channels(pcm.p,hw,2));check(snd_pcm_hw_params_set_rate(pcm.p,hw,rate,0));
 check(snd_pcm_hw_params_set_period_size(pcm.p,hw,period,0));
 check(snd_pcm_hw_params_set_periods(pcm.p,hw,2,0));
 check(snd_pcm_hw_params_set_buffer_size(pcm.p,hw,period*2));check(snd_pcm_hw_params(pcm.p,hw));
 snd_pcm_uframes_t actual_period,buffer;unsigned actual_rate,periods;int dir=0;
 check(snd_pcm_hw_params_get_period_size(hw,&actual_period,&dir));check(snd_pcm_hw_params_get_buffer_size(hw,&buffer));
 check(snd_pcm_hw_params_get_rate(hw,&actual_rate,&dir));check(snd_pcm_hw_params_get_periods(hw,&periods,&dir));
 if(actual_period!=period||buffer!=2*period||actual_rate!=rate||periods!=2)throw std::runtime_error("ALSA geometry differs from contract");
 snd_pcm_sw_params_t* sw;snd_pcm_sw_params_alloca(&sw);check(snd_pcm_sw_params_current(pcm.p,sw));
 check(snd_pcm_sw_params_set_avail_min(pcm.p,sw,period));
 check(snd_pcm_sw_params_set_start_threshold(pcm.p,sw,2*period));
 check(snd_pcm_sw_params_set_tstamp_mode(pcm.p,sw,SND_PCM_TSTAMP_ENABLE));
 check(snd_pcm_sw_params_set_tstamp_type(pcm.p,sw,SND_PCM_TSTAMP_TYPE_MONOTONIC));
 check(snd_pcm_sw_params(pcm.p,sw));check(snd_pcm_prepare(pcm.p));
 Mapping mapping("/cvdaw",true);Shared& s=mapping.get();Engine engine;
 std::array<float,max_period*2> output{};std::array<int32_t,max_period*2> integer{};
 // Warm DSP/libm and every backing page before FIFO. Restart frame timeline afterward.
 engine.render(s,output.data(),period,now());engine=Engine{};
 volatile unsigned char stack[65536]{};for(size_t i=0;i<sizeof(stack);i+=4096)stack[i]=0;
 cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(1,&cpus);
 if(sched_setaffinity(0,sizeof(cpus),&cpus))throw std::runtime_error("audio affinity");
 if(mlockall(MCL_CURRENT|MCL_FUTURE))throw std::runtime_error("mlockall: check LimitMEMLOCK");
 sched_param priority{};priority.sched_priority=80;
 if(sched_setscheduler(0,SCHED_FIFO,&priority))throw std::runtime_error("SCHED_FIFO: check LimitRTPRIO/CAP_SYS_NICE");
 signal(SIGINT,stop);signal(SIGTERM,stop);s.audio_ready.store(1,std::memory_order_release);
 int failure=0;
 while(running){
  // Wait for a writable period before rendering. ALSA blocking is confined to I/O.
  snd_pcm_sframes_t avail=snd_pcm_avail_update(pcm.p);
  if(avail<0){failure=int(avail);break;}
  if(avail<period){int r=snd_pcm_wait(pcm.p,100);if(r<0){if(r==-EINTR)continue;failure=r;break;}if(!r){failure=-ETIMEDOUT;break;}continue;}
  snd_pcm_sframes_t delay=0;int r=snd_pcm_delay(pcm.p,&delay);if(r<0){failure=r;break;}
  uint64_t begin=now();uint64_t dac=begin+uint64_t(std::max<snd_pcm_sframes_t>(0,delay))*1000000000/rate;
  engine.render(s,output.data(),period,dac);
  auto duration=now()-begin;auto maximum=s.max_render_ns.load(std::memory_order_relaxed);
  if(duration>maximum)s.max_render_ns.store(duration,std::memory_order_relaxed);
  if(format==SND_PCM_FORMAT_S32_LE)for(unsigned i=0;i<2*period;++i)integer[i]=int32_t(double(std::clamp(output[i],-1.f,1.f))*2147483647.0);
  unsigned written=0;
  while(written<period && running){
   const void* p=format==SND_PCM_FORMAT_FLOAT_LE?static_cast<const void*>(output.data()+2*written):static_cast<const void*>(integer.data()+2*written);
   auto n=snd_pcm_writei(pcm.p,p,period-written);
   if(n==-EINTR)continue;
   if(n<=0){failure=n?int(n):-EIO;break;}written+=unsigned(n);
  }
  if(failure)break;
  publish_clock(s,{engine.frame(),dac+uint64_t(period)*1000000000/rate});
 }
 if(failure==-EPIPE)s.xruns.fetch_add(1,std::memory_order_relaxed);
 s.audio_ready.store(0,std::memory_order_release);snd_pcm_drop(pcm.p);
 sched_param normal{};sched_setscheduler(0,SCHED_OTHER,&normal);munlockall();
 std::cerr<<"xruns="<<s.xruns.load()<<" drops="<<s.drops.load()<<" late="<<s.late.load()<<" scheduler_full="<<s.scheduler_full.load()<<" stem_starved_frames="<<s.stem_starved.load()<<" max_render_ns="<<s.max_render_ns.load()<<" max_capture_to_dac_ns="<<s.max_capture_to_dac_ns.load()<<'\n';
 if(failure){std::cerr<<"ALSA stopped: "<<snd_strerror(failure)<<'\n';return 2;}return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
