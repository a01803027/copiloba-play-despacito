#pragma once

#include "dsp_primitives.hpp"

namespace cv_daw {

struct StemFrame {
    std::array<Stereo, 4> stem{};
};

struct StemBus {
    SpscQueue<StemFrame, 8192> frames;
    std::atomic<float> requested_bpm{120.f};
    std::atomic<uint32_t> enabled{0};
};

struct RecordedHit {
    uint8_t note{};
    float velocity{};
};

struct PatternSlot {
    std::array<RecordedHit, 8> hits{};
    uint8_t count = 0;
};

class PendingCommands {
    struct Entry {
        AudioCommand command;
        uint64_t sequence;
    };

    std::array<Entry, 256> heap_{};
    uint32_t count_ = 0;
    uint64_t sequence_ = 0;

    static bool earlier(const Entry& a, const Entry& b) noexcept {
        return a.command.target_frame < b.command.target_frame
            || (a.command.target_frame == b.command.target_frame && a.sequence < b.sequence);
    }

public:
    bool push(AudioCommand command) noexcept {
        if (count_ == heap_.size()) {
            return false;
        }
        unsigned i = count_++;
        heap_[i] = {command, sequence_++};
        while (i) {
            unsigned parent = (i - 1) / 2;
            if (!earlier(heap_[i], heap_[parent])) {
                break;
            }
            std::swap(heap_[parent], heap_[i]);
            i = parent;
        }
        return true;
    }

    bool pop_due(uint64_t frame, AudioCommand& out) noexcept {
        if (!count_ || heap_[0].command.target_frame > frame) {
            return false;
        }
        out = heap_[0].command;
        heap_[0] = heap_[--count_];
        unsigned i = 0;
        while (2 * i + 1 < count_) {
            unsigned child = 2 * i + 1;
            if (child + 1 < count_ && earlier(heap_[child + 1], heap_[child])) {
                ++child;
            }
            if (!earlier(heap_[child], heap_[i])) {
                break;
            }
            std::swap(heap_[i], heap_[child]);
            i = child;
        }
        return true;
    }
};

class AudioEngine {
    static constexpr unsigned drum_kit_count = 6;

    SharedMemory& shared_;
    const Wavetables& tables_;
    std::array<SampleView, 4> samples_;
    StemBus& stems_;
    std::array<std::array<SampleView, 4>, drum_kit_count> drum_kits_{};

    std::array<VoicePool, 4> voices_{};
    std::array<std::array<CytomicSvf, 2>, 4> filters_{};
    std::array<std::array<OnePoleHpf, 2>, 2> hpf_{{
        {OnePoleHpf(210.f), OnePoleHpf(210.f)},
        {OnePoleHpf(240.f), OnePoleHpf(240.f)},
    }};

    StereoChorus pad_chorus_{};
    PresenceEq pad_eq_{1250.f, 0.85f, 0.34f};
    PresenceEq lead_eq_{1850.f, 0.85f, 0.46f};
    MonoSubCrossover mono_sub_{120.f};
    MasterGlueCompressor glue_{};

    std::array<OnePoleSmoother, 4> cutoff_{{
        OnePoleSmoother(8000, 18), OnePoleSmoother(8000, 18),
        OnePoleSmoother(8000, 18), OnePoleSmoother(8000, 18),
    }};
    std::array<OnePoleSmoother, 4> send_{};
    std::array<OnePoleSmoother, 4> bend_{};
    std::array<OnePoleSmoother, 4> gain_{{
        OnePoleSmoother(1, 16), OnePoleSmoother(1, 16),
        OnePoleSmoother(1, 16), OnePoleSmoother(1, 16),
    }};
    std::array<OnePoleSmoother, 4> pan_{{
        OnePoleSmoother(0, 16), OnePoleSmoother(0, 16),
        OnePoleSmoother(0, 16), OnePoleSmoother(0, 16),
    }};
    std::array<OnePoleSmoother, 4> volume_{{
        OnePoleSmoother(0.6f), OnePoleSmoother(0.6f),
        OnePoleSmoother(0.6f), OnePoleSmoother(0.6f),
    }};

    OnePoleSmoother bpm_{120.f, 50.f};
    OnePoleSmoother energy_{0.5f, 20.f};
    OnePoleSmoother delay_send_{0.f, 20.f};
    OnePoleSmoother resonance_{0.70710678f, 20.f};
    OnePoleSmoother sidechain_depth_{0.f, 20.f};

    float sidechain_peak_ = 0.f;
    float sidechain_env_ = 0.f;

    Reverb reverb_;
    StereoDelay delay_;
    PendingCommands pending_;

    std::array<std::array<PatternSlot, 32>, 4> patterns_{};
    std::array<bool, 4> muted_{};
    std::array<bool, 4> locked_{};
    std::array<bool, 4> stem_muted_{};
    std::array<uint8_t, 4> preset_{{0, 1, 2, 3}};

    uint64_t total_samples_rendered_ = 0;
    uint64_t tick_index_ = 0;
    uint64_t consumed_ = 0;
    uint64_t invalid_ = 0;
    uint64_t late_ = 0;
    uint64_t overflow_ = 0;
    uint64_t ticks_ = 0;
    uint64_t starved_ = 0;

    double phase_ = 0.0;
    bool first_tick_ = true;
    bool performance_ = false;
    double next_grid_ = 0.0;
    double grid_period_ = 6000.0;
    uint64_t current_grid_frame_ = 0;

    uint64_t next_grid_frame() const noexcept {
        return uint64_t(next_grid_ + 0.5);
    }

    void trigger(unsigned limb, uint8_t note, float velocity) noexcept {
        if (muted_[limb]) {
            return;
        }

        unsigned p = preset_[limb];
        SampleView asset{};
        bool is_drum_kit = false;

        if (limb == 3 && p >= 12 && p < 12 + drum_kit_count) {
            is_drum_kit = true;
            unsigned kit_idx = p - 12;
            unsigned idx = (note == 36) ? 0u : (note == 38) ? 1u : (note == 42 || note == 46) ? 2u : 3u;
            asset = drum_kits_[kit_idx][idx].samples ? drum_kits_[kit_idx][idx] : samples_[idx];
            if (idx == 0) {
                sidechain_peak_ = 1.f;
            }
        } else if (p == 12) {
            is_drum_kit = true;
            unsigned idx = (note == 36) ? 0u : (note == 38) ? 1u : (note == 42 || note == 46) ? 2u : 3u;
            asset = samples_[idx];
            if (idx == 0) {
                sidechain_peak_ = 1.f;
            }
        } else if (p >= 2 && p < 6) {
            asset = samples_[p - 2];
            if (performance_ && !asset.samples) {
                p = limb >= 2 ? 6 + (limb == 2) : p % 2;
            }
            if (p == 2 || p == 6) {
                sidechain_peak_ = 1.f;
            }
        } else if (p == 6) {
            sidechain_peak_ = 1.f;
        }

        voices_[limb].note_on(note, velocity, p, tables_, asset, is_drum_kit);
    }

    void execute(const AudioCommand& c) noexcept {
        unsigned l = unsigned(c.limb_id);
        if (c.event_type == EventType::STEM_MUTE) {
            if (l == 4 && c.note_or_param < 4) {
                stem_muted_[c.note_or_param] = !stem_muted_[c.note_or_param];
            } else {
                ++invalid_;
            }
            return;
        }
        if (l >= 4) {
            ++invalid_;
            return;
        }

        switch (c.event_type) {
            case EventType::NOTE_ON:
                if (c.note_or_param > 127 || c.value < 0.f || c.value > 1.f) {
                    ++invalid_;
                    break;
                }
                if (!locked_[l]) {
                    trigger(l, c.note_or_param, c.value);
                    if (!muted_[l]) {
                        bool nearest_future = performance_
                            ? (double(total_samples_rendered_) + grid_period_ / 2.0 >= next_grid_)
                            : (phase_ >= 0.5);
                        unsigned slot = unsigned((tick_index_ + (c.reserved == record_previous_grid ? 0 : nearest_future)) % 32);
                        auto& pattern = patterns_[l][slot];
                        if (pattern.count < pattern.hits.size()) {
                            pattern.hits[pattern.count++] = {c.note_or_param, c.value};
                        } else {
                            pattern.hits.back() = {c.note_or_param, c.value};
                        }
                    }
                }
                break;

            case EventType::NOTE_OFF:
                voices_[l].note_off(c.note_or_param);
                break;

            case EventType::MUTE_TOGGLE:
                muted_[l] = !muted_[l];
                if (muted_[l]) {
                    voices_[l].silence();
                }
                break;

            case EventType::LOOP_LOCK:
                locked_[l] = (c.value >= 0.5f);
                break;

            case EventType::PRESET_CHANGE:
                if (c.note_or_param < 12 + drum_kit_count) {
                    preset_[l] = c.note_or_param;
                } else {
                    ++invalid_;
                }
                break;

            case EventType::STEM_MUTE:
                break;
        }
    }

public:
    AudioEngine(
        SharedMemory& s,
        const Wavetables& t,
        std::array<SampleView, 4> samples,
        StemBus& stems,
        bool performance = false,
        const std::array<std::array<SampleView, 4>, drum_kit_count>* kits = nullptr
    )
        : shared_(s),
          tables_(t),
          samples_(samples),
          stems_(stems),
          performance_(performance) {
        if (kits) {
            drum_kits_ = *kits;
        } else {
            for (auto& k : drum_kits_) {
                k = samples;
            }
        }
    }

    uint64_t total_samples_rendered() const noexcept {
        return total_samples_rendered_;
    }

    double samples_per_sixteenth() const noexcept {
        return double(sample_rate) * 15.0 / bpm_.value();
    }

    // Pure bounded DSP and atomic memory operations. No clock calls, syscalls, allocators,
    // locks, library streaming/stretch calls, I/O, or waiting are reachable here.
    void processBlock(float* output, uint32_t frame_count) noexcept {
        if (!output || frame_count > block_frames) {
            return;
        }

        auto& controls = shared_.controls;
        std::array<float, 4> fc{}, rs{}, pb{}, sv{}, lg{}, pn{};
        for (unsigned l = 0; l < 4; ++l) {
            lg[l] = bounded(controls.limb_gain[l].load(std::memory_order_relaxed), 0.f, 1.f, 0.f);
            fc[l] = bounded(controls.limb_filter_cutoff[l].load(std::memory_order_relaxed), 20.f, 18000.f, 8000.f);
            rs[l] = bounded(controls.limb_reverb_send[l].load(std::memory_order_relaxed), 0.f, 1.f, 0.f);
            pb[l] = bounded(controls.limb_pitch_bend[l].load(std::memory_order_relaxed), -12.f, 12.f, 0.f);
            sv[l] = bounded(controls.stem_volumes[l].load(std::memory_order_relaxed), 0.f, 1.f, 0.f);
            pn[l] = bounded(controls.limb_pan[l].load(std::memory_order_relaxed), -1.f, 1.f, 0.f);
        }

        float bpm = bounded(controls.target_bpm.load(std::memory_order_relaxed), 60.f, 200.f, 120.f);
        float energy = bounded(controls.groove_energy.load(std::memory_order_relaxed), 0.f, 1.f, 0.5f);
        float d_send = bounded(controls.delay_send.load(std::memory_order_relaxed), 0.f, 1.f, 0.f);
        float res = bounded(controls.filter_resonance.load(std::memory_order_relaxed), 0.5f, 3.5f, 0.70710678f);
        float sc = bounded(controls.sidechain_depth.load(std::memory_order_relaxed), 0.f, 0.85f, 0.f);
        stems_.requested_bpm.store(bpm, std::memory_order_relaxed);

        AudioCommand command;
        for (unsigned i = 0; i < command_capacity && shared_.commands.try_pop(command); ++i) {
            ++consumed_;
            if (!valid(command)) {
                ++invalid_;
                continue;
            }
            if (command.target_frame == 0) {
                command.target_frame = total_samples_rendered_;
            }
            if (command.target_frame < total_samples_rendered_) {
                ++late_;
            }
            // Reject absurd future timestamps rather than allowing one producer to reserve
            // the entire pending pool indefinitely. Ten seconds of scheduling is supported.
            if (command.target_frame > total_samples_rendered_ + 10 * sample_rate) {
                ++invalid_;
                continue;
            }
            if (!pending_.push(command)) {
                ++overflow_;
            }
        }

        bool stems_enabled = (stems_.enabled.load(std::memory_order_acquire) == 1);

        for (unsigned n = 0; n < frame_count; ++n, ++total_samples_rendered_) {
            float current_bpm = bpm_.next(bpm);
            float current_energy = energy_.next(energy);

            bool tick = performance_
                ? (total_samples_rendered_ >= next_grid_frame())
                : (first_tick_ || phase_ >= 1.0 - 1e-12);
            if (tick) {
                if (first_tick_) {
                    first_tick_ = false;
                } else {
                    phase_ = std::max(0.0, phase_ - 1.0);
                    ++tick_index_;
                }
                if (performance_) {
                    current_grid_frame_ = total_samples_rendered_;
                    grid_period_ = double(sample_rate) * 15.0 / bpm;
                    next_grid_ += grid_period_;
                }
            }

            while (pending_.pop_due(total_samples_rendered_, command)) {
                execute(command);
            }

            if (tick) {
                ++ticks_;
                for (unsigned l = 0; l < 4; ++l) {
                    if (locked_[l]) {
                        auto& slot = patterns_[l][tick_index_ % 32];
                        for (unsigned j = 0; j < slot.count; ++j) {
                            trigger(l, slot.hits[j].note, slot.hits[j].velocity);
                        }
                    }
                }
            }

            Stereo master{};
            Stereo reverb_in{};
            Stereo delay_in{};

            float cur_delay = delay_send_.next(d_send);
            float cur_res = resonance_.next(res);
            float cur_sc = sidechain_depth_.next(sc);

            // Click-free smooth sidechain pump: ~2.6 ms smooth attack into ~150 ms release.
            sidechain_peak_ *= 0.99986f;
            sidechain_env_ += 0.008f * (sidechain_peak_ - sidechain_env_);

            for (unsigned l = 0; l < 4; ++l) {
                float cutoff = cutoff_[l].next(fc[l]);
                float send = send_[l].next(rs[l]);
                float bend = tables_.bend(bend_[l].next(pb[l]));

                auto bus = voices_[l].next(tables_, bend);
                float bus_res = (l < 2) ? cur_res : 0.70710678f;
                filters_[l][0].coefficients(cutoff, bus_res);
                filters_[l][1].coefficients(cutoff, bus_res);
                bus.l = filters_[l][0].next(bus.l).low;
                bus.r = filters_[l][1].next(bus.r).low;

                // Carve sub-bass mud + add studio presence EQ & chorus on studio synth presets (8..14).
                if (l == 0 && preset_[l] >= 8) {
                    bus.l = hpf_[0][0].next(bus.l);
                    bus.r = hpf_[0][1].next(bus.r);
                    bus = pad_eq_.next(pad_chorus_.next(bus, 0.26f));
                } else if (l == 1 && preset_[l] >= 8) {
                    bus.l = hpf_[1][0].next(bus.l);
                    bus.r = hpf_[1][1].next(bus.r);
                    bus = lead_eq_.next(bus);
                }

                float limb_gain = gain_[l].next(lg[l]);
                if (l == 0 && cur_sc > 1e-4f) {
                    limb_gain *= (1.f - cur_sc * sidechain_env_);
                }
                bus = apply_pan({bus.l * limb_gain, bus.r * limb_gain}, pan_[l].next(pn[l]));

                master.l += bus.l;
                master.r += bus.r;
                reverb_in.l += bus.l * send;
                reverb_in.r += bus.r * send;
                if (l < 2 && cur_delay > 1e-4f) {
                    delay_in.l += bus.l * cur_delay;
                    delay_in.r += bus.r * cur_delay;
                }
            }

            auto del = delay_.next(delay_in, current_bpm);
            master.l += del.l * 0.36f;
            master.r += del.r * 0.36f;
            reverb_in.l += del.l * 0.20f;
            reverb_in.r += del.r * 0.20f;

            auto wet = reverb_.next(reverb_in);
            master.l += wet.l * 0.38f;
            master.r += wet.r * 0.38f;

            StemFrame stems{};
            if (stems_enabled && !stems_.frames.try_pop(stems)) {
                ++starved_;
            }
            for (unsigned l = 0; l < 4; ++l) {
                float vol = volume_[l].next(stem_muted_[l] ? 0.f : sv[l]);
                master.l += stems.stem[l].l * vol;
                master.r += stems.stem[l].r * vol;
            }

            // Studio master bus chain: tight mono-sub (<120 Hz) + gentle SSL glue compression.
            if (preset_[0] >= 8 || preset_[1] >= 8 || preset_[2] >= 8 || preset_[3] >= 12) {
                master = glue_.next(mono_sub_.next(master, 1.14f));
            }

            float master_gain = 0.62f + 0.28f * current_energy;
            output[2 * n] = master.l * master_gain;
            output[2 * n + 1] = master.r * master_gain;

            if (!performance_) {
                phase_ += current_bpm / (sample_rate * 15.0);
            }
        }

        limit_stereo(output, frame_count);

        auto& t = shared_.telemetry;
        t.commands_consumed.store(consumed_, std::memory_order_relaxed);
        t.invalid_commands.store(invalid_, std::memory_order_relaxed);
        t.late_commands.store(late_, std::memory_order_relaxed);
        t.scheduler_overflows.store(overflow_, std::memory_order_relaxed);
        t.sixteenth_ticks.store(ticks_, std::memory_order_relaxed);
        t.stem_starved_frames.store(starved_, std::memory_order_relaxed);
        if (performance_) {
            uint64_t next_frame = next_grid_frame();
            t.grid_snapshot.store((next_frame << 16) | (next_frame - current_grid_frame_), std::memory_order_release);
        }
        t.total_samples_rendered.store(total_samples_rendered_, std::memory_order_release);
    }
};

}  // namespace cv_daw
