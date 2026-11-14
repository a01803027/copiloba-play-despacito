#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace cvdaw {
constexpr unsigned rate=48000, max_period=256, limbs=4;
constexpr uint32_t abi=0x43564404;
static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<float>::is_always_lock_free);
// Cross-process atomics require the same ABI/toolchain and lock-free ARM64 atomics.
template<class T, uint32_t N> struct Spsc {
 static_assert(N && !(N&(N-1)));
 alignas(64) std::atomic<uint32_t> write{0};
 alignas(64) std::atomic<uint32_t> read{0};
 alignas(64) std::array<T,N> data{};
 bool push(const T& value) noexcept {
  auto w=write.load(std::memory_order_relaxed);
  if(w-read.load(std::memory_order_acquire)==N) return false;
  data[w&(N-1)]=value; write.store(w+1,std::memory_order_release); return true;
 }
 bool pop(T& value) noexcept {
  auto r=read.load(std::memory_order_relaxed);
  if(r==write.load(std::memory_order_acquire)) return false;
  value=data[r&(N-1)]; read.store(r+1,std::memory_order_release); return true;
 }
};
enum class Op:uint8_t { note, mute, preset, panic };
enum class Mode:uint8_t { live, muted, looping };
struct Command {
 uint64_t frame{};
 uint64_t capture_ns{};
 Op op{Op::note}; uint8_t limb{}, note{60}, value{};
 float velocity{0.8f};
};
struct Stereo {float l{},r{};};
struct alignas(64) Controls {
 std::array<std::atomic<float>,4> cutoff{{8000,8000,8000,8000}};
 std::atomic<float> energy{0}, bpm{120}, stem_gain{0.35f};
 std::array<std::atomic<float>,4> groove{{0,0,0,0}};
 std::atomic<uint64_t> heartbeat_ns{0};
};
struct Shared {
 std::atomic<uint32_t> magic{0};
 std::atomic<uint32_t> audio_ready{0}, brain_ready{0}, stems_ready{0};
 Controls controls;
 Spsc<Command,1024> commands;
 // One worker mixes all four stems; exactly one producer and one consumer.
 Spsc<Stereo,16384> stem_audio;
 alignas(64) std::atomic<uint64_t> clock_seq{0}, frame{0}, dac_ns{0};
 std::atomic<uint64_t> xruns{0}, drops{0}, late{0}, stem_starved{0}, scheduler_full{0};
 std::atomic<uint64_t> max_render_ns{0}, max_capture_to_dac_ns{0};
};
struct Clock {uint64_t frame{}, ns{};};
inline Clock clock_snapshot(const Shared& s) noexcept {
 // Bounded retry: caller skips this vision frame if writer was interrupted.
 for(unsigned i=0;i<8;++i){
  auto a=s.clock_seq.load(std::memory_order_acquire);
  if(a&1) continue;
  Clock c{s.frame.load(std::memory_order_relaxed),s.dac_ns.load(std::memory_order_relaxed)};
  std::atomic_thread_fence(std::memory_order_acquire);
  if(a==s.clock_seq.load(std::memory_order_acquire)) return c;
 }
 return {};
}
inline void publish_clock(Shared& s, Clock c) noexcept {
 s.clock_seq.fetch_add(1,std::memory_order_acq_rel);
 s.frame.store(c.frame,std::memory_order_relaxed);
 s.dac_ns.store(c.ns,std::memory_order_relaxed);
 s.clock_seq.fetch_add(1,std::memory_order_release);
}
class Mapping {
 int fd_=-1; Shared* p_=nullptr; bool owner_=false; std::string name_;
public:
 Mapping(const char* name,bool create):owner_(create),name_(name){
  fd_=shm_open(name,create ? O_RDWR|O_CREAT|O_EXCL : O_RDWR,0600);
  if(fd_<0) throw std::runtime_error("shm_open failed (audio must start first; stale objects require explicit cleanup)");
  if(create && ftruncate(fd_,sizeof(Shared))<0){close(fd_);shm_unlink(name);throw std::runtime_error("ftruncate");}
  struct stat st{};
  if(fstat(fd_,&st)<0 || st.st_size<static_cast<off_t>(sizeof(Shared))){close(fd_);if(create)shm_unlink(name);throw std::runtime_error("shared memory size mismatch");}
  void* ptr=mmap(nullptr,sizeof(Shared),PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0);
  if(ptr==MAP_FAILED){close(fd_);if(create)shm_unlink(name);throw std::runtime_error("mmap");}
  p_=static_cast<Shared*>(ptr);
  if(create){new(p_) Shared{};p_->magic.store(abi,std::memory_order_release);}
  else if(p_->magic.load(std::memory_order_acquire)!=abi){munmap(p_,sizeof(Shared));close(fd_);throw std::runtime_error("shared memory ABI/not initialized");}
 }
 ~Mapping(){if(p_)munmap(p_,sizeof(Shared));if(fd_>=0)close(fd_);if(owner_)shm_unlink(name_.c_str());}
 Mapping(const Mapping&)=delete; Mapping& operator=(const Mapping&)=delete;
 Shared& get() noexcept{return *p_;}
};
}
