#pragma once
#include <cstdint>
namespace cv_daw {
// User-space architectural counter: no clock_gettime/mach calls in callback.
inline uint64_t hardware_ticks()noexcept{
#if defined(__aarch64__)
 uint64_t value;asm volatile("isb\n\tmrs %0, cntvct_el0":"=r"(value)::"memory");return value;
#else
 return 0; // Timing unavailable on non-AArch64 hosts; DSP still works.
#endif
}
inline uint64_t hardware_frequency()noexcept{
#if defined(__aarch64__)
 uint64_t value;asm volatile("mrs %0, cntfrq_el0":"=r"(value));return value;
#else
 return 0;
#endif
}
}
