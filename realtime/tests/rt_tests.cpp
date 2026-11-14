#include "audio_engine.hpp"
#include "rt_startup.hpp"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <sys/wait.h>
#ifdef __linux__
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#endif
using namespace cv_daw;
bool cv_daw_test_forbid_runtime_calls=false;
#define no_allocations cv_daw_test_forbid_runtime_calls
void* operator new(std::size_t n){if(no_allocations)std::abort();if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete[](void* p)noexcept{::operator delete(p);}
void* operator new(std::size_t n,std::align_val_t align){if(no_allocations)std::abort();void* p=nullptr;if(posix_memalign(&p,static_cast<std::size_t>(align),n?n:1))throw std::bad_alloc();return p;}
void* operator new[](std::size_t n,std::align_val_t align){return ::operator new(n,align);}
void operator delete(void* p,std::align_val_t)noexcept{std::free(p);}
void operator delete[](void* p,std::align_val_t)noexcept{std::free(p);}
#ifdef __GNUC__
#if !defined(__clang__)
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
#endif
#endif
#define CHECK(x) do{if(!(x)){std::cerr<<__LINE__<<": " #x "\n";std::exit(1);}}while(false)
int main(){
 flush_subnormals();
 {CommandQueue queue;AudioCommand c;for(unsigned i=0;i<256;++i)CHECK(queue.try_push({LimbID::L_ARM,EventType::NOTE_ON,uint8_t(i%128),0,1,i}));CHECK(!queue.try_push({}));
  for(unsigned i=0;i<256;++i){CHECK(queue.try_pop(c));CHECK(c.target_frame==i);}CHECK(!queue.try_pop(c));
  queue.read_index.store(UINT32_MAX-100);queue.write_index.store(UINT32_MAX-100);
  for(unsigned i=0;i<256;++i)CHECK(queue.try_push({LimbID::L_ARM,EventType::NOTE_ON,60,0,1,i}));
  for(unsigned i=0;i<256;++i){CHECK(queue.try_pop(c));CHECK(c.target_frame==i);}}
 {SpscQueue<uint32_t,256> queue;std::thread producer([&]{for(unsigned i=0;i<200000;++i)while(!queue.try_push(i))std::this_thread::yield();});
  for(unsigned i=0;i<200000;++i){unsigned value;while(!queue.try_pop(value))std::this_thread::yield();CHECK(value==i);}producer.join();}
 {auto name=std::string("/cv-daw-test-")+std::to_string(getpid());SharedMemoryMapping owner(true,name.c_str());
  auto child=fork();CHECK(child>=0);if(child==0){SharedMemoryMapping attached(false,name.c_str());bool ok=attached.get().commands.try_push({LimbID::R_LEG,EventType::NOTE_ON,70,0,.9f,999});_exit(ok?0:1);}
  int status;CHECK(waitpid(child,&status,0)==child && WIFEXITED(status)&&WEXITSTATUS(status)==0);AudioCommand c;CHECK(owner.get().commands.try_pop(c));CHECK(c.target_frame==999&&c.limb_id==LimbID::R_LEG);}
 {OnePoleSmoother smoother(0,10);float last=0;for(unsigned i=0;i<1000;++i){float y=smoother.next(1);CHECK(y>=last&&y<=1);last=y;}CHECK(last>.8f);}
 {for(float cutoff:{20.f,400.f,8000.f,18000.f}){CytomicSvf filter;filter.coefficients(cutoff,4);for(unsigned i=0;i<20000;++i){auto v=filter.next(i==0?1.f:0.f);CHECK(finite(v.low)&&finite(v.band)&&finite(v.high));CHECK(std::abs(v.low)<10);}}}
 {for(float input:{-1e30f,-3.f,-1.f,0.f,1.f,3.f,1e30f,std::bit_cast<float>(0x7fc00000u),std::bit_cast<float>(0x7f800000u)}){float y=soft_clip(input);CHECK(finite(y)&&y>=-1&&y<=1);}}
 auto tables=std::make_unique<Wavetables>();auto shared=std::make_unique<SharedMemory>();auto stems=std::make_unique<StemBus>();
 auto engine=std::make_unique<AudioEngine>(*shared,*tables,std::array<SampleView,4>{},*stems);
 std::array<float,256> output{};
 CHECK(shared->commands.try_push({LimbID::L_ARM,EventType::NOTE_ON,60,0,1,64}));
 no_allocations=true;engine->processBlock(output.data(),128);no_allocations=false;
 for(unsigned i=0;i<128;++i)CHECK(output[i]==0);bool sound=false;for(unsigned i=128;i<256;++i)sound|=std::abs(output[i])>1e-6f;CHECK(sound);
 CHECK(shared->telemetry.commands_consumed.load()==1);
 CHECK(shared->commands.try_push({LimbID::L_ARM,EventType::LOOP_LOCK,0,0,1,128}));
 // At 120 BPM, the same recorded slot recurs after exactly 192000 frames.
 float before=0,after=0;
 while(engine->total_samples_rendered()<192128){auto frame=engine->total_samples_rendered();engine->processBlock(output.data(),128);
  for(unsigned n=0;n<128;++n){if(frame+n>=191872 && frame+n<192000)before=std::max(before,std::abs(output[2*n]));if(frame+n>=192000)after=std::max(after,std::abs(output[2*n]));}}
 CHECK(before<1e-6f&&after>1e-4f);CHECK(shared->telemetry.sixteenth_ticks.load()==33);
 {auto local=std::make_unique<SharedMemory>();auto bus=std::make_unique<StemBus>();AudioEngine e(*local,*tables,{},*bus);
  local->controls.target_bpm.store(200);local->controls.limb_filter_cutoff[0].store(std::bit_cast<float>(0x7fc00000u));
  for(unsigned i=0;i<1000;++i){no_allocations=true;e.processBlock(output.data(),128);no_allocations=false;for(float value:output)CHECK(finite(value)&&std::abs(value)<=1);}
  CHECK(std::abs(e.samples_per_sixteenth()-3600)<.1);}
 {PendingCommands pending;for(unsigned i=0;i<256;++i)CHECK(pending.push({LimbID::L_ARM,EventType::NOTE_ON,60,0,1,256-i}));CHECK(!pending.push({}));AudioCommand c;for(unsigned i=1;i<=256;++i)CHECK(pending.pop_due(i,c)&&c.target_frame==i);}
 {PendingCommands pending;for(unsigned i=0;i<32;++i)CHECK(pending.push({LimbID::L_ARM,EventType::NOTE_ON,uint8_t(i),0,1,9}));AudioCommand c;for(unsigned i=0;i<32;++i)CHECK(pending.pop_due(9,c)&&c.note_or_param==i);}
 {std::array<float,10000> sample{};sample.fill(.5f);Voice voice;voice.start(60,1,2,*tables,{sample.data(),uint32_t(sample.size()),1});
  for(unsigned i=0;i<1000;++i)voice.next(*tables,1);CHECK(voice.active);voice.off();for(unsigned i=0;i<6000;++i)voice.next(*tables,1);CHECK(!voice.active);}
 {auto local=std::make_unique<SharedMemory>();auto bus=std::make_unique<StemBus>();bus->enabled.store(1);StemFrame f;f.stem[0]={1000,-1000};for(unsigned i=0;i<128;++i)CHECK(bus->frames.try_push(f));AudioEngine e(*local,*tables,{},*bus);e.processBlock(output.data(),128);
  for(unsigned i=0;i<128;++i){CHECK(output[2*i]<=1&&output[2*i]>=-1);CHECK(output[2*i+1]<=1&&output[2*i+1]>=-1);}CHECK(local->telemetry.stem_starved_frames.load()==0);}
 {auto local=std::make_unique<SharedMemory>();auto bus=std::make_unique<StemBus>();AudioEngine e(*local,*tables,{},*bus);
  local->commands.try_push({LimbID::L_ARM,EventType::NOTE_ON,60,0,1,0});
  local->controls.limb_gain[0].store(0);
  for(unsigned i=0;i<100;++i){no_allocations=true;e.processBlock(output.data(),128);no_allocations=false;}
  for(float value:output)CHECK(std::abs(value)<1e-6f);
 }
#ifdef __linux__
 // Kernel-enforced zero-system-call test, including libc/vDSO-fallback paths.
 // All allocation, libc warmup and fork happen BEFORE installing the filter.
 {pid_t child=fork();CHECK(child>=0);if(child==0){
   sock_filter filter[]={BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(seccomp_data,nr)),
    BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,SYS_exit_group,2,0),BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,SYS_exit,1,0),
    BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW)};
   sock_fprog program{static_cast<unsigned short>(std::size(filter)),filter};
   if(prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)||prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&program))_exit(77);
   no_allocations=true;for(unsigned i=0;i<100;++i)engine->processBlock(output.data(),128);_exit(0);
  }int status;CHECK(waitpid(child,&status,0)==child);CHECK(WIFEXITED(status)&&WEXITSTATUS(status)==0);}
#endif
 std::cout<<"Modules 4/5 tests passed\n";
}
