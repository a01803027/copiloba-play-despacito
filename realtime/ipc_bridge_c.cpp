#include "ipc_bridge.hpp"

#include <algorithm>

using namespace cv_daw;

extern "C" {

uint32_t cv_daw_abi_size() {
    return sizeof(SharedMemory);
}

uint32_t cv_daw_command_size() {
    return sizeof(AudioCommand);
}

int cv_daw_attach(void* memory, uint32_t bytes) {
    if (!memory
        || reinterpret_cast<uintptr_t>(memory) % alignof(SharedMemory) != 0
        || bytes < sizeof(SharedMemory)) {
        return 0;
    }
    auto& s = *static_cast<SharedMemory*>(memory);
    return SharedMemoryMapping::compatible(s)
        && s.header.ready.load(std::memory_order_acquire) == 1;
}

int cv_daw_claim(void* memory, uint32_t token) {
    if (!memory || !token) {
        return 0;
    }
    uint32_t expected = 0;
    return static_cast<SharedMemory*>(memory)->header.producer_claim.compare_exchange_strong(
        expected, token, std::memory_order_acq_rel
    );
}

void cv_daw_release(void* memory, uint32_t token) {
    auto& claim = static_cast<SharedMemory*>(memory)->header.producer_claim;
    claim.compare_exchange_strong(token, 0, std::memory_order_acq_rel);
}

int cv_daw_push(void* memory, const void* packed_command, uint32_t token) {
    auto& s = *static_cast<SharedMemory*>(memory);
    if (!token || s.header.producer_claim.load(std::memory_order_acquire) != token) {
        return -1;
    }

    AudioCommand c;
    std::memcpy(&c, packed_command, sizeof(c));
    if (!valid(c)) {
        return -2;
    }

    if (s.commands.try_push(c)) {
        return 1;
    }

    auto drops = s.telemetry.producer_drops.load(std::memory_order_relaxed);
    s.telemetry.producer_drops.store(drops + 1, std::memory_order_relaxed);
    return 0;
}

// Parameter groups:
//   0 = cutoff, 1 = reverb, 2 = bend, 3 = BPM, 4 = energy, 5 = stem volumes,
//   6 = gain, 7 = pan, 8 = delay, 9 = resonance, 10 = sidechain.
int cv_daw_control(void* memory, uint32_t group, uint32_t index, float value, uint32_t token) {
    auto& s = *static_cast<SharedMemory*>(memory);
    if (!token
        || s.header.producer_claim.load(std::memory_order_acquire) != token
        || !finite(value)) {
        return 0;
    }
    if (index >= 4) {
        return 0;
    }

    auto& c = s.controls;
    switch (group) {
        case 0:
            c.limb_filter_cutoff[index].store(std::clamp(value, 20.f, 18000.f), std::memory_order_relaxed);
            break;
        case 1:
            c.limb_reverb_send[index].store(std::clamp(value, 0.f, 1.f), std::memory_order_relaxed);
            break;
        case 2:
            c.limb_pitch_bend[index].store(std::clamp(value, -12.f, 12.f), std::memory_order_relaxed);
            break;
        case 3:
            c.target_bpm.store(std::clamp(value, 60.f, 200.f), std::memory_order_relaxed);
            break;
        case 4:
            c.groove_energy.store(std::clamp(value, 0.f, 1.f), std::memory_order_relaxed);
            break;
        case 5:
            c.stem_volumes[index].store(std::clamp(value, 0.f, 1.f), std::memory_order_relaxed);
            break;
        case 6:
            c.limb_gain[index].store(std::clamp(value, 0.f, 1.f), std::memory_order_relaxed);
            break;
        case 7:
            c.limb_pan[index].store(std::clamp(value, -1.f, 1.f), std::memory_order_relaxed);
            break;
        case 8:
            c.delay_send.store(std::clamp(value, 0.f, 1.f), std::memory_order_relaxed);
            break;
        case 9:
            c.filter_resonance.store(std::clamp(value, 0.5f, 3.5f), std::memory_order_relaxed);
            break;
        case 10:
            c.sidechain_depth.store(std::clamp(value, 0.f, 0.85f), std::memory_order_relaxed);
            break;
        default:
            return 0;
    }
    return 1;
}

uint64_t cv_daw_metric(void* memory, uint32_t index) {
    auto& t = static_cast<SharedMemory*>(memory)->telemetry;
    switch (index) {
        case 0:  return t.total_samples_rendered.load(std::memory_order_acquire);
        case 1:  return t.commands_consumed.load(std::memory_order_relaxed);
        case 2:  return t.producer_drops.load(std::memory_order_relaxed);
        case 3:  return t.invalid_commands.load(std::memory_order_relaxed);
        case 4:  return t.late_commands.load(std::memory_order_relaxed);
        case 5:  return t.scheduler_overflows.load(std::memory_order_relaxed);
        case 6:  return t.sixteenth_ticks.load(std::memory_order_relaxed);
        case 7:  return t.stem_starved_frames.load(std::memory_order_relaxed);
        case 8:  return t.xruns.load(std::memory_order_relaxed);
        case 9:  return t.callback_count.load(std::memory_order_relaxed);
        case 10: return t.callback_sum_ticks.load(std::memory_order_relaxed);
        case 11: return t.callback_max_ticks.load(std::memory_order_relaxed);
        case 12: return t.counter_frequency.load(std::memory_order_relaxed);
        case 13: return t.grid_snapshot.load(std::memory_order_acquire);
        default: return 0;
    }
}

}  // extern "C"
