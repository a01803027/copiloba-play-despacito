#include "cvdaw/motion.hpp"
#include "cvdaw/music.hpp"
#include "cvdaw/dsp.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <memory>
#include <new>
#include <unistd.h>
#include <sys/wait.h>
static bool forbid_allocation=false;
void* operator new(std::size_t n){if(forbid_allocation)throw std::bad_alloc();if(void* p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete[](void* p)noexcept{::operator delete(p);}
using namespace cvdaw;
#define REQUIRE(x) do {if(!(x)){std::cerr<<__LINE__<<": " #x "\n";std::exit(1);}}while(false)
int main(){
 {Spsc<uint32_t,4> q;uint32_t n;for(unsigned i=0;i<4;++i)REQUIRE(q.push(i));REQUIRE(!q.push(5));for(unsigned i=0;i<4;++i){REQUIRE(q.pop(n));REQUIRE(n==i);}REQUIRE(!q.pop(n));
  q.read.store(UINT32_MAX-2);q.write.store(UINT32_MAX-2);for(unsigned i=0;i<4;++i)REQUIRE(q.push(i));for(unsigned i=0;i<4;++i){REQUIRE(q.pop(n));REQUIRE(n==i);}}
 {Spsc<uint32_t,1024> q;std::thread producer([&]{for(unsigned i=0;i<200000;++i)while(!q.push(i))std::this_thread::yield();});
  for(unsigned i=0;i<200000;++i){unsigned n;while(!q.pop(n))std::this_thread::yield();REQUIRE(n==i);}producer.join();}
 {auto q=quantize(48000+480,48000,120);REQUIRE(q.audible==48480);REQUIRE(q.grid==48000);
  auto early=quantize(48000+6000-240,48000,120);REQUIRE(early.audible==54000);
  auto strict=quantize(48000+6000-1440,48000,120);REQUIRE(strict.audible==52560);
  auto full=quantize(52560,48000,120,false);REQUIRE(full.audible==54000);
  REQUIRE(pitch(-100)==60);REQUIRE(pitch(100)==75);}
 {Euro f;REQUIRE(f.step(1,.016)==1);REQUIRE(f.step(2,.016)>1);f.reset();REQUIRE(f.step(3,.016)==3);}
 {Pose p{};p[11]={-.2,.2,0};p[12]={.2,.2,0};p[23]={-.2,.6,0};p[24]={.2,.6,0};p[15]={-.4,.1,0};p[16]={.4,.1,0};p[27]={-.2,1,0};p[28]={.2,1,0};
  Motion a,b;MotionFrame x,y;Pose transformed=p;for(auto& v:transformed)v=v*2+Vec{3,2,1};
  REQUIRE(a.process(p,1000000000,x));REQUIRE(b.process(transformed,1000000000,y));for(unsigned i=0;i<4;++i)REQUIRE((x.position[i]-y.position[i]).norm()<1e-9);
  REQUIRE(!a.process(p,1000000000,x));for(auto hit:x.hit)REQUIRE(!hit);
  REQUIRE(a.process(p,1200000000,x));for(auto hit:x.hit)REQUIRE(!hit);}
 {Pose p{};p[11]={-.2,.2,0};p[12]={.2,.2,0};p[23]={-.2,.6,0};p[24]={.2,.6,0};
  p[15]={-.4,.1,0};p[16]={.4,.1,0};p[27]={-.2,1,0};p[28]={.2,1,0};Motion m;m.min_speed=.1;m.deceleration=.1;
  unsigned hits=0;uint64_t last=0;MotionFrame f;
  for(unsigned i=0;i<180;++i){p[15].x=-.4+(i%60<15?double(i%60)*.025:.375);uint64_t ns=1000000000+uint64_t(i)*16666667;
   REQUIRE(m.process(p,ns,f));if(f.hit[0]){REQUIRE(!last || ns-last>=80000000);last=ns;++hits;}}
  REQUIRE(hits>=3);
  Pose degenerate{};m.reset();REQUIRE(!m.process(degenerate,5000000000,f));
 }
 {Scheduler scheduler;REQUIRE(scheduler.add({50}));REQUIRE(scheduler.add({10}));REQUIRE(scheduler.add({30}));Command c;REQUIRE(!scheduler.due(9,c));REQUIRE(scheduler.due(10,c)&&c.frame==10);REQUIRE(scheduler.due(100,c)&&c.frame==30);REQUIRE(scheduler.due(100,c)&&c.frame==50);}
 {auto s=std::make_unique<Shared>();publish_clock(*s,{123,456});auto c=clock_snapshot(*s);REQUIRE(c.frame==123&&c.ns==456);
  s->controls.heartbeat_ns.store(1000000000);Engine engine;std::array<float,512> out{};
  REQUIRE(s->commands.push({64,999000000,Op::note,0,60,0,1}));forbid_allocation=true;engine.render(*s,out.data(),256,1000000000);forbid_allocation=false;
  for(unsigned i=0;i<128;++i)REQUIRE(out[i]==0);bool sound=false;for(unsigned i=128;i<512;++i){REQUIRE(std::isfinite(out[i]));REQUIRE(std::abs(out[i])<=1);sound|=std::abs(out[i])>1e-6;}REQUIRE(sound);REQUIRE(s->max_capture_to_dac_ns.load()>0);
  engine.command({0,0,Op::panic});engine.render(*s,out.data(),256,1300000000);for(auto v:out)REQUIRE(std::isfinite(v));}
 {auto s=std::make_unique<Shared>();Brain brain(*s);std::array<bool,4> hit{true,false,false,false};std::array<double,4> height{};std::array<uint8_t,4> g{};
  brain.step(48000,1,hit,height,g,120);Command c;REQUIRE(s->commands.pop(c));REQUIRE(c.op==Op::note&&c.limb==0);
  g[0]=1;brain.step(49000,2,hit,height,g,120);REQUIRE(s->commands.pop(c)&&c.op==Op::mute&&c.value==1);REQUIRE(!s->commands.pop(c));}
 {auto name=std::string("/cvdaw-test-")+std::to_string(getpid());Mapping parent(name.c_str(),true);
  pid_t pid=fork();REQUIRE(pid>=0);
  if(pid==0){Mapping child(name.c_str(),false);bool ok=child.get().commands.push({77,0,Op::note,2,65,0,1});_exit(ok?0:1);}
  int status=0;REQUIRE(waitpid(pid,&status,0)==pid);REQUIRE(WIFEXITED(status)&&WEXITSTATUS(status)==0);Command c;REQUIRE(parent.get().commands.pop(c)&&c.frame==77&&c.limb==2);
 }
 {Scheduler scheduler;for(unsigned i=0;i<1024;++i)REQUIRE(scheduler.add({i}));REQUIRE(!scheduler.add({2000}));}
 {auto s=std::make_unique<Shared>();Brain brain(*s);std::array<bool,4> hit{true,false,false,false};std::array<double,4> h{};std::array<uint8_t,4> g{};
  brain.step(48000,1,hit,h,g,120);Command c;REQUIRE(s->commands.pop(c));hit.fill(false);g[0]=2;
  brain.step(48100,2,hit,h,g,120);REQUIRE(s->commands.pop(c)&&c.op==Op::mute);
  brain.step(48000+32*6000,3,hit,h,g,120);REQUIRE(s->commands.pop(c)&&c.op==Op::note&&c.frame==240000);
 }
 std::cout<<"core tests passed\n";
}
