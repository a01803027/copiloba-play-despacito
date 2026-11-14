#include "ipc_bridge.hpp"
#include <algorithm>
using namespace cv_daw;
extern "C" {
uint32_t cv_daw_abi_size(){return sizeof(SharedMemory);}
uint32_t cv_daw_command_size(){return sizeof(AudioCommand);}
int cv_daw_attach(void* memory,uint32_t bytes){
 if(!memory || reinterpret_cast<uintptr_t>(memory)%alignof(SharedMemory) || bytes<sizeof(SharedMemory))return 0;
 auto& s=*static_cast<SharedMemory*>(memory);return SharedMemoryMapping::compatible(s)&&s.header.ready.load(std::memory_order_acquire)==1;
}
int cv_daw_claim(void* memory,uint32_t token){
 if(!memory || !token)return 0;
 uint32_t expected=0;return static_cast<SharedMemory*>(memory)->header.producer_claim.compare_exchange_strong(expected,token,std::memory_order_acq_rel);
}
void cv_daw_release(void* memory,uint32_t token){
 auto& a=static_cast<SharedMemory*>(memory)->header.producer_claim;a.compare_exchange_strong(token,0,std::memory_order_acq_rel);
}
int cv_daw_push(void* memory,const void* packed_command,uint32_t token){
 auto& s=*static_cast<SharedMemory*>(memory);
 if(s.header.producer_claim.load(std::memory_order_acquire)!=token || !token)return -1;
 AudioCommand c;std::memcpy(&c,packed_command,sizeof(c));if(!valid(c))return -2;
 if(s.commands.try_push(c))return 1;
 auto n=s.telemetry.producer_drops.load(std::memory_order_relaxed);s.telemetry.producer_drops.store(n+1,std::memory_order_relaxed);return 0;
}
// Parameter groups: 0=cutoff,1=reverb,2=bend,3=BPM,4=energy,5=stem volumes.
int cv_daw_control(void* memory,uint32_t group,uint32_t index,float value,uint32_t token){
 auto& s=*static_cast<SharedMemory*>(memory);
 if(!token || s.header.producer_claim.load(std::memory_order_acquire)!=token || !finite(value))return 0;
 auto& c=s.controls;
 if(index>=4)return 0;
 switch(group){
 case 0:c.limb_filter_cutoff[index].store(std::clamp(value,20.f,18000.f),std::memory_order_relaxed);break;
 case 1:c.limb_reverb_send[index].store(std::clamp(value,0.f,1.f),std::memory_order_relaxed);break;
 case 2:c.limb_pitch_bend[index].store(std::clamp(value,-12.f,12.f),std::memory_order_relaxed);break;
 case 3:c.target_bpm.store(std::clamp(value,60.f,200.f),std::memory_order_relaxed);break;
 case 4:c.groove_energy.store(std::clamp(value,0.f,1.f),std::memory_order_relaxed);break;
 case 5:c.stem_volumes[index].store(std::clamp(value,0.f,1.f),std::memory_order_relaxed);break;
 default:return 0;
 }return 1;
}
uint64_t cv_daw_metric(void* memory,uint32_t index){
 auto& s=*static_cast<SharedMemory*>(memory);auto& t=s.telemetry;
 switch(index){
 case 0:return t.total_samples_rendered.load(std::memory_order_acquire);
 case 1:return t.commands_consumed.load(std::memory_order_relaxed);
 case 2:return t.producer_drops.load(std::memory_order_relaxed);
 case 3:return t.invalid_commands.load(std::memory_order_relaxed);
 case 4:return t.late_commands.load(std::memory_order_relaxed);
 case 5:return t.scheduler_overflows.load(std::memory_order_relaxed);
 case 6:return t.sixteenth_ticks.load(std::memory_order_relaxed);
 case 7:return t.stem_starved_frames.load(std::memory_order_relaxed);
 case 8:return t.xruns.load(std::memory_order_relaxed);
 default:return 0;
 }
}
}
