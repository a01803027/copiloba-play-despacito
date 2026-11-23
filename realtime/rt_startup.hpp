#pragma once
#include <cerrno>
#include <cstdint>
#include <pthread.h>
#include <sys/mman.h>
#ifdef __linux__
#include <sched.h>
#endif
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace cv_daw {
inline int lock_process_memory()noexcept{
#ifdef __linux__
 return mlockall(MCL_CURRENT|MCL_FUTURE)==0?0:errno;
#else
 return ENOTSUP; // macOS live playback uses CoreAudio, not Linux RT policy.
#endif
}
inline void flush_subnormals()noexcept{
#if defined(__aarch64__)
 uint64_t fpcr;asm volatile("mrs %0, fpcr":"=r"(fpcr));
 // ARMv8-A FZ flushes single/double denormal inputs AND outputs. Cortex-A35 has
 // neither the newer FIZ bit nor a separate x86 DAZ bit. Preserve other FP state.
 fpcr|=uint64_t(1)<<24;asm volatile("msr fpcr, %0\n\tisb"::"r"(fpcr):"memory");
#elif defined(__SSE__)
 _mm_setcsr(_mm_getcsr()|(1u<<15)|(1u<<6));
#endif
}
// Call from the miniaudio worker's backend START hook, before ALSA prefill can
// invoke dataCallback. Nothing in this function is called by processBlock().
inline int initialize_audio_thread()noexcept{
#ifdef __linux__
 cpu_set_t cpuset;CPU_ZERO(&cpuset);CPU_SET(1,&cpuset);
 int error=pthread_setaffinity_np(pthread_self(),sizeof(cpuset),&cpuset);if(error)return error;
 // Touch sufficient worker stack before raising FIFO priority.
 volatile unsigned char prefault[128*1024]{};
 for(unsigned i=0;i<sizeof(prefault);i+=4096)prefault[i]=0;
 flush_subnormals();sched_param scheduling{};scheduling.sched_priority=90;
 return pthread_setschedparam(pthread_self(),SCHED_FIFO,&scheduling);
#else
 flush_subnormals();return ENOTSUP;
#endif
}
}
