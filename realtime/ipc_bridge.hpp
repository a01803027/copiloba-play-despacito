#pragma once
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cv_daw {
inline constexpr char shm_name[]="/cv_daw_shm";
inline constexpr uint32_t ipc_magic=0x43564457, ipc_version=2, command_capacity=256;
static_assert(std::endian::native==std::endian::little);
static_assert(std::atomic<uint32_t>::is_always_lock_free && sizeof(std::atomic<uint32_t>)==4);
static_assert(std::atomic<uint64_t>::is_always_lock_free && sizeof(std::atomic<uint64_t>)==8);
static_assert(std::atomic<float>::is_always_lock_free && sizeof(std::atomic<float>)==4);
enum class LimbID:uint8_t {L_ARM=0,R_ARM=1,L_LEG=2,R_LEG=3,MASTER=4};
enum class EventType:uint8_t {NOTE_ON=0,NOTE_OFF=1,MUTE_TOGGLE=2,LOOP_LOCK=3,PRESET_CHANGE=4,STEM_MUTE=5};
struct alignas(16) AudioCommand {
 LimbID limb_id{}; EventType event_type{}; uint8_t note_or_param{},reserved{};
 float value{};uint64_t target_frame{};
};
static_assert(sizeof(AudioCommand)==16 && alignof(AudioCommand)==16);
static_assert(offsetof(AudioCommand,value)==4 && offsetof(AudioCommand,target_frame)==8);
static_assert(std::is_trivially_copyable_v<AudioCommand>);

// Single producer and single consumer only. Fixed instruction count; no CAS loops.
template<typename T,uint32_t N> struct alignas(64) SpscQueue {
 static_assert(N && (N&(N-1))==0 && N<(1u<<31));
 alignas(64) std::atomic<uint32_t> write_index{0};
 alignas(64) std::atomic<uint32_t> read_index{0};
 alignas(64) std::array<T,N> slots{};
 bool try_push(const T& command) noexcept {
  auto w=write_index.load(std::memory_order_relaxed);
  if(w-read_index.load(std::memory_order_acquire)==N)return false;
  slots[w&(N-1)]=command;write_index.store(w+1,std::memory_order_release);return true;
 }
 bool try_pop(T& command) noexcept {
  auto r=read_index.load(std::memory_order_relaxed);
  if(r==write_index.load(std::memory_order_acquire))return false;
  command=slots[r&(N-1)];read_index.store(r+1,std::memory_order_release);return true;
 }
 uint32_t size_for_producer()const noexcept {
  return write_index.load(std::memory_order_relaxed)-read_index.load(std::memory_order_acquire);
 }
};
using CommandQueue=SpscQueue<AudioCommand,command_capacity>;
struct alignas(64) AtomicControlBlock {
 std::array<std::atomic<float>,4> limb_filter_cutoff{{8000,8000,8000,8000}};
 std::array<std::atomic<float>,4> limb_reverb_send{{0,0,0,0}};
 std::array<std::atomic<float>,4> limb_pitch_bend{{0,0,0,0}}; // semitones [-12,+12]
 std::atomic<float> target_bpm{120},groove_energy{0.5f};
 std::array<std::atomic<float>,4> stem_volumes{{0.6f,0.6f,0.6f,0.6f}};
 std::array<std::atomic<float>,4> limb_gain{{1,1,1,1}};
};
struct alignas(64) IpcHeader {
 std::atomic<uint32_t> magic{0};
 uint32_t version=ipc_version,bytes=0,command_bytes=sizeof(AudioCommand),capacity=command_capacity;
 uint32_t controls_offset=0,queue_offset=0,telemetry_offset=0;
 std::atomic<uint32_t> ready{0},producer_claim{0};
};
struct alignas(64) Telemetry {
 std::atomic<uint64_t> total_samples_rendered{0},commands_consumed{0},producer_drops{0},invalid_commands{0};
 std::atomic<uint64_t> late_commands{0},scheduler_overflows{0},sixteenth_ticks{0},stem_starved_frames{0};
 std::atomic<uint64_t> xruns{0},rendered_peak_bits{0};
 std::atomic<uint64_t> callback_count{0},callback_sum_ticks{0},callback_max_ticks{0},counter_frequency{0};
};
struct alignas(64) SharedMemory {
 IpcHeader header;AtomicControlBlock controls;CommandQueue commands;Telemetry telemetry;
};
static_assert(sizeof(IpcHeader)==64 && sizeof(AtomicControlBlock)==128);
static_assert(offsetof(SharedMemory,controls)==64 && offsetof(SharedMemory,commands)==192);
static_assert(offsetof(CommandQueue,write_index)==0 && offsetof(CommandQueue,read_index)==64 && offsetof(CommandQueue,slots)==128);
static_assert(sizeof(CommandQueue)==4224 && offsetof(SharedMemory,telemetry)==4416);
static_assert(sizeof(SharedMemory)==4544);
// Bit inspection remains valid with -ffast-math (std::isfinite may be optimized away).
inline bool finite(float x)noexcept{return (std::bit_cast<uint32_t>(x)&0x7f800000u)!=0x7f800000u;}
inline bool valid(const AudioCommand& c)noexcept {
 return unsigned(c.limb_id)<=4 && unsigned(c.event_type)<=5 && c.reserved==0 && finite(c.value);
}
class SharedMemoryMapping {
 int fd_=-1;SharedMemory* p_=nullptr;bool owner_=false;std::string name_;
 [[noreturn]] static void fail(const char* message){throw std::runtime_error(std::string(message)+": "+std::strerror(errno));}
public:
 explicit SharedMemoryMapping(bool create,const char* name=shm_name):owner_(create),name_(name){
  fd_=shm_open(name,create?O_RDWR|O_CREAT|O_EXCL:O_RDWR,0660);if(fd_<0)fail("shm_open");
  if(create && ftruncate(fd_,sizeof(SharedMemory))<0){int saved=errno;close(fd_);shm_unlink(name);errno=saved;fail("ftruncate");}
  struct stat st{};
  if(fstat(fd_,&st)<0 || st.st_size<static_cast<off_t>(sizeof(SharedMemory))){close(fd_);if(create)shm_unlink(name);throw std::runtime_error("IPC size mismatch");}
  void* p=mmap(nullptr,sizeof(SharedMemory),PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0);
  if(p==MAP_FAILED){int saved=errno;close(fd_);if(create)shm_unlink(name);errno=saved;fail("mmap");}p_=static_cast<SharedMemory*>(p);
  if(create){
   new(p_)SharedMemory{};auto& h=p_->header;h.bytes=sizeof(SharedMemory);
   h.controls_offset=offsetof(SharedMemory,controls);h.queue_offset=offsetof(SharedMemory,commands);h.telemetry_offset=offsetof(SharedMemory,telemetry);
   h.magic.store(ipc_magic,std::memory_order_release);
  }else if(!compatible(*p_)){
   munmap(p_,sizeof(SharedMemory));close(fd_);throw std::runtime_error("IPC version/initialization mismatch");
  }
 }
 static bool compatible(const SharedMemory& s)noexcept {
  return s.header.magic.load(std::memory_order_acquire)==ipc_magic && s.header.version==ipc_version && s.header.bytes==sizeof(SharedMemory) && s.header.command_bytes==sizeof(AudioCommand) &&
   s.header.capacity==command_capacity && s.header.controls_offset==offsetof(SharedMemory,controls) &&
   s.header.queue_offset==offsetof(SharedMemory,commands) && s.header.telemetry_offset==offsetof(SharedMemory,telemetry);
 }
 ~SharedMemoryMapping(){if(p_)munmap(p_,sizeof(SharedMemory));if(fd_>=0)close(fd_);if(owner_)shm_unlink(name_.c_str());}
 SharedMemoryMapping(const SharedMemoryMapping&)=delete;
 SharedMemoryMapping& operator=(const SharedMemoryMapping&)=delete;
 SharedMemory& get()noexcept{return *p_;}
};
}
