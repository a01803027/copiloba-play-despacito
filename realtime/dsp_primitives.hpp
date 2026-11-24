#pragma once

#include "ipc_bridge.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace cv_daw {

inline constexpr uint32_t sample_rate = 48000;
inline constexpr uint32_t block_frames = 128;
inline constexpr uint32_t voices_per_limb = 12;

inline float bounded(float value, float lo, float hi, float fallback) noexcept {
    return finite(value) ? std::clamp(value, lo, hi) : fallback;
}

struct Stereo {
    float l{};
    float r{};

    Stereo operator+(Stereo b) const noexcept {
        return {l + b.l, r + b.r};
    }
};

inline Stereo apply_pan(Stereo in, float pan) noexcept {
    float p = std::clamp(pan, -1.f, 1.f);
    float left_gain = std::sqrt(0.5f * (1.f - p));
    float right_gain = std::sqrt(0.5f * (1.f + p));
    // Preserve mono center loudness (~1.0 when pan == 0).
    constexpr float norm = 1.41421356f;
    return {in.l * left_gain * norm, in.r * right_gain * norm};
}

class OnePoleSmoother {
    float state_;
    float alpha_;

public:
    OnePoleSmoother() : OnePoleSmoother(0.f, 10.f) {}

    explicit OnePoleSmoother(float initial, float milliseconds = 10.f)
        : state_(initial),
          alpha_(1.f - std::exp(-1.f / (sample_rate * milliseconds * 0.001f))) {}

    float next(float target) noexcept {
        float updated = state_ + alpha_ * (target - state_);
        state_ = (updated == state_) ? target : updated;
        return state_;
    }

    float value() const noexcept {
        return state_;
    }
};

class OnePoleHpf {
    float prev_in_ = 0.f;
    float prev_out_ = 0.f;
    float r_ = 0.975f;

public:
    explicit OnePoleHpf(float hz = 190.f) {
        float rc = 1.f / (2.f * float(std::numbers::pi) * hz);
        float dt = 1.f / float(sample_rate);
        r_ = rc / (rc + dt);
    }

    float next(float in) noexcept {
        float out = r_ * (prev_out_ + in - prev_in_);
        prev_in_ = in;
        prev_out_ = out;
        return out;
    }
};

class Adsr {
    enum class Stage : uint8_t { off, attack, decay, sustain, release };

    Stage stage_ = Stage::off;
    float level_ = 0.f;
    float attack_step_ = 1.f / 144.f;
    float decay_coeff_ = 0.999f;
    float sustain_ = 0.55f;
    float release_coeff_ = 0.9984f;

public:
    void configure(float attack_ms, float decay_ms, float sustain, float release_ms) noexcept {
        float a_samp = std::max(1.f, sample_rate * attack_ms * 0.001f);
        float d_samp = std::max(1.f, sample_rate * decay_ms * 0.001f);
        float r_samp = std::max(1.f, sample_rate * release_ms * 0.001f);
        sustain_ = std::clamp(sustain, 0.f, 1.f);
        attack_step_ = 1.f / a_samp;
        // Exponential curves (e^-4.6 ~= 0.01 over decay; e^-9.21 ~= 1e-4 over release -> natural analog tail).
        decay_coeff_ = std::exp(-4.605f / d_samp);
        release_coeff_ = std::exp(-9.210f / r_samp);
    }

    void on() noexcept {
        stage_ = Stage::attack;
        level_ = 0.f;
    }

    void off() noexcept {
        if (stage_ != Stage::off && stage_ != Stage::release) {
            stage_ = Stage::release;
        }
    }

    bool active() const noexcept {
        return stage_ != Stage::off;
    }

    bool releasing() const noexcept {
        return stage_ == Stage::release;
    }

    float next() noexcept {
        switch (stage_) {
            case Stage::attack:
                level_ += attack_step_;
                if (level_ >= 1.f) {
                    level_ = 1.f;
                    stage_ = Stage::decay;
                }
                break;

            case Stage::decay:
                level_ = sustain_ + (level_ - sustain_) * decay_coeff_;
                if (std::abs(level_ - sustain_) <= 1e-4f) {
                    level_ = sustain_;
                    stage_ = Stage::sustain;
                }
                break;

            case Stage::sustain:
                break;

            case Stage::release:
                level_ *= release_coeff_;
                if (level_ <= 1.5e-4f) {
                    level_ = 0.f;
                    stage_ = Stage::off;
                }
                break;

            case Stage::off:
                break;
        }
        return level_;
    }
};

struct SampleView {
    const float* samples = nullptr;
    uint32_t frames = 0;
    uint32_t channels = 1;
};

struct Wavetables {
    static constexpr uint32_t size = 2048;
    static constexpr uint32_t bands = 10;

    alignas(16) std::array<std::array<float, size + 1>, bands> saw{};
    alignas(16) std::array<std::array<float, size + 1>, bands> tri{};
    alignas(16) std::array<float, size + 1> sine{};
    std::array<float, 128> increments{};
    std::array<float, 1025> bend_ratios{};

    Wavetables() {
        for (uint32_t i = 0; i < size; ++i) {
            double phase = 2.0 * std::numbers::pi * i / size;
            sine[i] = float(std::sin(phase));

            for (uint32_t b = 0; b < bands; ++b) {
                double sum_saw = 0.0;
                double sum_tri = 0.0;
                uint32_t harmonics = 512u >> b;

                for (uint32_t h = 1; h <= harmonics; ++h) {
                    sum_saw -= std::sin(phase * h) / h;
                    if (h & 1u) {
                        double sign = ((h >> 1) & 1u) ? -1.0 : 1.0;
                        sum_tri += sign * std::sin(phase * h) / (double(h) * double(h));
                    }
                }
                saw[b][i] = float(sum_saw * 2.0 / std::numbers::pi);
                tri[b][i] = float(sum_tri * 8.0 / (std::numbers::pi * std::numbers::pi));
            }
        }

        sine[size] = sine[0];
        for (auto& b : saw) {
            b[size] = b[0];
        }
        for (auto& b : tri) {
            b[size] = b[0];
        }
        for (unsigned n = 0; n < 128; ++n) {
            increments[n] = float(440.0 * std::exp2((double(n) - 69.0) / 12.0) / sample_rate);
        }
        for (unsigned i = 0; i < bend_ratios.size(); ++i) {
            bend_ratios[i] = float(std::exp2((-12.0 + 24.0 * i / 1024.0) / 12.0));
        }
    }

    float bend(float semitones) const noexcept {
        float x = (semitones + 12.f) * (1024.f / 24.f);
        unsigned i = std::min(1023u, unsigned(x));
        return bend_ratios[i] + (bend_ratios[i + 1] - bend_ratios[i]) * (x - i);
    }

    float oscillator(float phase, unsigned band, bool is_sine) const noexcept {
        float x = phase * size;
        unsigned i = unsigned(x);
        float frac = x - i;
        const float* p = is_sine ? sine.data() : saw[band].data();
        return p[i] + (p[i + 1] - p[i]) * frac;
    }

    float triangle(float phase, unsigned band) const noexcept {
        float x = phase * size;
        unsigned i = unsigned(x);
        float frac = x - i;
        const float* p = tri[band].data();
        return p[i] + (p[i + 1] - p[i]) * frac;
    }

    unsigned band_for(float increment) const noexcept {
        // Reserve an octave of headroom for the full +12-semitone bend range.
        unsigned b = 0;
        while (b + 1 < bands && (512u >> b) * increment * 2.f > 0.45f) {
            ++b;
        }
        return b;
    }
};

// TPT/Cytomic SVF. Positive g and k preserve topology stability for every supported cutoff.
class CytomicSvf {
    float ic1_ = 0.f;
    float ic2_ = 0.f;
    float a1_ = 1.f;
    float a2_ = 0.f;
    float a3_ = 0.f;
    float k_ = 1.41421356f;

public:
    void coefficients(float cutoff, float resonance = 0.70710678f) noexcept {
        float x = float(std::numbers::pi) * cutoff / sample_rate;
        float x2 = x * x;
        // [5/4] rational tan approximation, x in [0.0013, 1.1781].
        float g = x * (105.f - 10.f * x2) / (105.f - 45.f * x2 + x2 * x2);
        k_ = 1.f / std::clamp(resonance, 0.5f, 4.f);
        a1_ = 1.f / (1.f + g * (g + k_));
        a2_ = g * a1_;
        a3_ = g * a2_;
    }

    struct Outputs {
        float low;
        float band;
        float high;
    };

    Outputs next(float in) noexcept {
        float v3 = in - ic2_;
        float v1 = a1_ * ic1_ + a2_ * v3;
        float v2 = ic2_ + a2_ * ic1_ + a3_ * v3;
        ic1_ = 2.f * v1 - ic1_;
        ic2_ = 2.f * v2 - ic2_;
        return {v2, v1, in - k_ * v1 - v2};
    }
};

struct Voice {
    Adsr envelope;
    SampleView sample{};
    CytomicSvf vfilt_l{};
    CytomicSvf vfilt_r{};

    float phase = 0.f;
    float phase2 = 0.f;
    float phase3 = 0.f;
    float increment = 0.f;
    float gain = 0.f;
    float mod_env = 0.f;

    float filt_env = 0.f;
    float filt_decay = 0.9995f;
    float filt_base = 2400.f;
    float filt_depth = 0.f;
    float filt_res = 0.85f;

    double sample_position = 0.0;
    uint32_t age = 0;
    uint32_t max_age = 2 * sample_rate;
    uint32_t noise = 1;

    uint8_t note = 0;
    uint8_t band = 0;
    uint8_t preset_id = 0;
    uint8_t drum = 0;

    bool active = false;
    bool is_sample = false;
    bool is_sine = false;
    bool use_vfilt = false;

    void start(
        uint8_t n,
        float velocity,
        unsigned preset,
        const Wavetables& tables,
        SampleView asset,
        bool is_drum_kit = false
    ) noexcept {
        note = n;
        phase = 0.f;
        phase2 = 0.17f;
        phase3 = 0.61f;
        sample_position = 0.0;
        age = 0;
        increment = tables.increments[n];
        band = uint8_t(tables.band_for(increment));
        gain = velocity;
        preset_id = uint8_t(preset);
        drum = (preset == 6 || preset == 7) ? uint8_t(preset - 5) : 0;
        noise = 1234567u + uint32_t(n) * 2654435761u;

        bool is_crash = (is_drum_kit && n == 49);
        bool is_riser = (!is_drum_kit && preset >= 8 && n == 120);
        is_sample = (preset >= 2 && preset < 6) || (is_drum_kit && !is_crash);
        is_sine = (preset == 1);
        sample = asset;
        active = !is_sample || (sample.samples && sample.frames);

        mod_env = 1.0f;
        filt_env = 1.0f;
        use_vfilt = false;
        vfilt_l = {};
        vfilt_r = {};

        // Configure expressive studio envelopes & per-voice filter envelopes (presets 0..7 unchanged).
        if (is_riser) {
            // Dedicated Cinematic Soundwave Riser (triggered by MIDI 120 on Double-Arm Rise).
            // Smooth 1.8s upward pitch & resonant filter sweep with wide stereo air shimmer.
            envelope.configure(180.f, 1500.f, 0.85f, 450.f);
            max_age = uint32_t(2.4f * sample_rate);
            increment = tables.increments[48];
            band = uint8_t(tables.band_for(increment));
            use_vfilt = true;
            filt_base = 360.f;
            filt_depth = 8800.f;
            filt_decay = 0.9999f;
            filt_res = 2.25f;
            filt_env = 0.0f;
            mod_env = 0.0f;
            phase3 = 0.0f;
        } else if (is_drum_kit) {
            if (n == 42) {
                envelope.configure(1.2f, 48.f, 0.04f, 45.f);
                max_age = uint32_t(0.10f * sample_rate);
            } else if (n == 46) {
                envelope.configure(1.2f, 200.f, 0.15f, 210.f);
                max_age = uint32_t(0.32f * sample_rate);
            } else if (n == 49) {
                envelope.configure(3.f, 520.f, 0.03f, 320.f);
                max_age = uint32_t(0.80f * sample_rate);
                phase3 = 0.f;
                mod_env = 0.f;
            } else {
                envelope.configure(1.2f, 180.f, 0.70f, 140.f);
                max_age = uint32_t(0.65f * sample_rate);
            }
        } else if (preset == 8) {
            // Warm Analog Unison Pad (blooming filter envelope)
            envelope.configure(95.f, 550.f, 0.76f, 680.f);
            max_age = uint32_t(4.0f * sample_rate);
            use_vfilt = true;
            filt_base = 950.f;
            filt_depth = 2400.f * velocity;
            filt_decay = 0.99993f;
            filt_res = 0.82f;
            filt_env = 0.15f;
        } else if (preset == 9) {
            // Studio Rhodes / FM Electric Piano
            envelope.configure(3.5f, 620.f, 0.28f, 320.f);
            max_age = uint32_t(3.0f * sample_rate);
            use_vfilt = true;
            filt_base = 1150.f;
            filt_depth = 3600.f * velocity;
            filt_decay = 0.99972f;
            filt_res = 0.80f;
        } else if (preset == 10) {
            // Prophet / Diva Supersaw Lead
            envelope.configure(3.5f, 240.f, 0.36f, 210.f);
            max_age = 2 * sample_rate;
            use_vfilt = true;
            filt_base = 1300.f;
            filt_depth = 5200.f * velocity;
            filt_decay = 0.99968f;
            filt_res = 1.05f;
        } else if (preset == 11) {
            // Moog-Style Punchy Sub + Mid Bass
            envelope.configure(4.f, 180.f, 0.60f, 120.f);
            max_age = 2 * sample_rate;
            use_vfilt = true;
            filt_base = 240.f;
            filt_depth = 1450.f * velocity;
            filt_decay = 0.99942f;
            filt_res = 1.25f;
        } else if (preset == 13) {
            // Melodic House / Anjunadeep Pluck
            envelope.configure(3.f, 195.f, 0.16f, 175.f);
            max_age = 2 * sample_rate;
            use_vfilt = true;
            filt_base = 580.f;
            filt_depth = 5400.f * velocity;
            filt_decay = 0.99932f;
            filt_res = 1.22f;
        } else if (preset == 14) {
            // Organic Acoustic Bell-Pluck (Marimba / Nylon hybrid)
            envelope.configure(3.f, 280.f, 0.08f, 210.f);
            max_age = 2 * sample_rate;
            use_vfilt = true;
            filt_base = 850.f;
            filt_depth = 4400.f * velocity;
            filt_decay = 0.99928f;
            filt_res = 0.90f;
        } else {
            envelope.configure(3.f, 80.f, 0.55f, 120.f);
            max_age = 2 * sample_rate;
        }

        if (use_vfilt) {
            float c0 = std::clamp(filt_base + filt_depth * filt_env, 120.f, 13500.f);
            vfilt_l.coefficients(c0, filt_res);
            vfilt_r.coefficients(c0, filt_res);
        }
        envelope.on();
    }

    void off() noexcept {
        envelope.off();
    }

    Stereo next(const Wavetables& tables, float bend_ratio) noexcept {
        if (!active) {
            return {};
        }

        float env = envelope.next();
        Stereo out{};

        if (is_sample) {
            uint32_t i = uint32_t(sample_position);
            if (i >= sample.frames) {
                active = false;
                return {};
            }
            uint32_t j = std::min(i + 1, sample.frames - 1);
            float frac = float(sample_position - i);
            auto read = [&](unsigned channel) {
                unsigned c = (sample.channels == 1) ? 0 : channel;
                float a = sample.samples[i * sample.channels + c];
                float b = sample.samples[j * sample.channels + c];
                return a + (b - a) * frac;
            };
            out = {read(0) * gain * env, read(1) * gain * env};
            sample_position += bend_ratio;
            if (!envelope.active()) {
                active = false;
            }
        } else if (drum) {
            constexpr unsigned duration = sample_rate / 5;
            float tail = 1.f - float(age) / duration;
            noise ^= noise << 13;
            noise ^= noise >> 17;
            noise ^= noise << 5;
            float tone = tables.oscillator(phase, band, true);
            float white = float(noise >> 8) * (2.f / 16777216.f) - 1.f;
            float value = (drum == 1 ? tone : 0.75f * white + 0.25f * tone) * tail * tail * gain * 0.55f;
            out = {value, value};
            phase += increment * bend_ratio * (1.f + tail * tail * (drum == 1 ? 3.f : 1.f));
            phase -= unsigned(phase);
            if (age + 1 == duration) {
                active = false;
            }
        } else if (preset_id >= 8 && note == 120 && use_vfilt) {
            // Studio Rising Soundwave: lush 3-voice stereo pitch glide (+1 octave over ~1.75s)
            // + band-limited warm air swell.
            float progress = std::min(1.f, float(age) / (1.75f * float(sample_rate)));
            filt_env = progress * progress;          // exponential upward filter sweep 360 Hz -> 9160 Hz
            float glide = 1.f + progress * progress; // smooth +1 octave upward pitch rise
            float inc = std::min(increment * glide * bend_ratio, 0.45f);
            unsigned b = tables.band_for(inc);

            noise ^= noise << 13;
            noise ^= noise >> 17;
            noise ^= noise << 5;
            float w = float(noise >> 8) * (2.f / 16777216.f) - 1.f;
            phase3 += 0.28f * (w - phase3);  // warm band-limited air shimmer (zero harsh hiss)

            float s_l = tables.oscillator(phase, b, false);
            float s_r = tables.oscillator(phase2, b, false);
            float air = phase3 * (0.15f + 0.35f * progress);
            float swell = (0.25f + 0.75f * progress) * env * gain * 0.26f;
            out = {
                (s_l * 0.80f + s_r * 0.20f + air) * swell,
                (s_r * 0.80f + s_l * 0.20f - air * 0.85f) * swell,
            };

            phase += std::min(inc * 0.993f, 0.45f);
            if (phase >= 1.f) phase -= 1.f;
            phase2 += std::min(inc * 1.007f, 0.45f);
            if (phase2 >= 1.f) phase2 -= 1.f;
        } else if (preset_id >= 12 && note == 49 && !use_vfilt) {
            // Warm Studio Crash Cymbal: band-limited silky shimmer (no raw white-noise hiss).
            noise ^= noise << 13;
            noise ^= noise >> 17;
            noise ^= noise << 5;
            float w1 = float(noise >> 8) * (2.f / 16777216.f) - 1.f;

            noise ^= noise << 13;
            noise ^= noise >> 17;
            noise ^= noise << 5;
            float w2 = float(noise >> 8) * (2.f / 16777216.f) - 1.f;

            phase3 += 0.40f * (w1 - phase3);
            mod_env += 0.40f * (w2 - mod_env);

            float metal = tables.oscillator(phase, band, true) * tables.oscillator(phase2, band, true);
            float l_sig = phase3 * 0.36f + metal * 0.34f;
            float r_sig = mod_env * 0.36f - metal * 0.34f;
            float scale = env * gain * 0.24f;
            out = {l_sig * scale, r_sig * scale};

            phase += 0.072f;
            if (phase >= 1.f) phase -= 1.f;
            phase2 += 0.113f;
            if (phase2 >= 1.f) phase2 -= 1.f;
        } else if (preset_id == 8) {
            // Warm Analog Unison Pad: lush stereo detuned saw/tri blend with gentle blooming filter.
            float inc = std::min(increment * bend_ratio, 0.45f);
            filt_env += 0.00018f * (1.0f - filt_env);
            float saw_mix = 0.52f + 0.28f * gain;
            float tri_mix = 1.f - saw_mix;

            float s_c = tri_mix * tables.triangle(phase, band) + saw_mix * tables.oscillator(phase, band, false);
            float s_l = saw_mix * tables.oscillator(phase2, band, false) + tri_mix * tables.triangle(phase2, band);
            float s_r = saw_mix * tables.oscillator(phase3, band, false) + tri_mix * tables.triangle(phase3, band);
            float scale = env * gain * 0.22f;
            out = {
                (s_c * 0.40f + s_l * 0.86f + s_r * 0.14f) * scale,
                (s_c * 0.40f + s_r * 0.86f + s_l * 0.14f) * scale,
            };

            phase += inc;
            if (phase >= 1.f) phase -= 1.f;
            phase2 += std::min(inc * 0.9945f, 0.45f);  // -9.5 cents
            if (phase2 >= 1.f) phase2 -= 1.f;
            phase3 += std::min(inc * 1.0055f, 0.45f);  // +9.5 cents
            if (phase3 >= 1.f) phase3 -= 1.f;
        } else if (preset_id == 9) {
            // Studio Rhodes / FM Electric Piano: expressive bell tine attack + rich warm harmonic body.
            float inc = std::min(increment * bend_ratio, 0.45f);
            mod_env *= 0.99976f;
            filt_env *= filt_decay;

            float vel_mod = 0.65f + 0.90f * gain;
            float mod = tables.oscillator(phase2, band, true) * (0.12f + 1.45f * vel_mod * mod_env * mod_env);
            float p_mod = phase + mod * 0.18f;
            p_mod -= std::floor(p_mod);

            float body = tables.oscillator(p_mod, band, true);
            float shimmer = tables.oscillator(phase3, band, true) * 0.24f * vel_mod * mod_env;
            float scale = env * gain * 0.30f;
            out = {(body + shimmer) * scale, (body - shimmer * 0.82f) * scale};

            phase += inc;
            if (phase >= 1.f) phase -= 1.f;
            phase2 += std::min(inc * 2.0f, 0.45f);
            if (phase2 >= 1.f) phase2 -= 1.f;
            phase3 += std::min(inc * 3.004f, 0.45f);
            if (phase3 >= 1.f) phase3 -= 1.f;
        } else if (preset_id == 10) {
            // Prophet / Diva Supersaw Lead: lush stereo detuned saws + warm triangle core + filter pluck.
            float inc = std::min(increment * bend_ratio, 0.45f);
            mod_env *= 0.99970f;
            filt_env *= filt_decay;

            float sl = tables.oscillator(phase2, band, false);
            float sr = tables.oscillator(phase3, band, false);
            float body = tables.triangle(phase, band) * 0.36f;
            float bright = 0.55f + 0.45f * mod_env;
            float scale = env * gain * 0.26f;
            out = {
                (body + (sl * 0.86f + sr * 0.14f) * bright) * scale,
                (body + (sr * 0.86f + sl * 0.14f) * bright) * scale,
            };

            phase += inc;
            if (phase >= 1.f) phase -= 1.f;
            phase2 += std::min(inc * 0.9942f, 0.45f);
            if (phase2 >= 1.f) phase2 -= 1.f;
            phase3 += std::min(inc * 1.0058f, 0.45f);
            if (phase3 >= 1.f) phase3 -= 1.f;
        } else if (preset_id == 11) {
            // Moog-Style Sub + Mid Bass: tight sine foundation + saturated resonant saw pluck.
            float inc = std::min(increment * bend_ratio, 0.45f);
            filt_env *= filt_decay;

            float sub = tables.oscillator(phase, band, true);
            float saw = tables.oscillator(phase2, band, false);
            float warm_saw = std::tanh(1.6f * saw) * (0.38f + 0.25f * gain);
            float v = (sub * 0.72f + warm_saw) * env * gain * 0.28f;
            out = {v, v};

            phase += inc;
            if (phase >= 1.f) phase -= 1.f;
            phase2 += std::min(inc * 1.0012f, 0.45f);  // subtle analog phasing warmth
            if (phase2 >= 1.f) phase2 -= 1.f;
        } else if (preset_id == 13) {
            // Melodic House / Anjunadeep Pluck: wide detuned saws + warm triangle + snappy filter envelope.
            float inc = std::min(increment * bend_ratio, 0.45f);
            mod_env *= 0.99925f;
            filt_env *= filt_decay;
            unsigned oct_band = tables.band_for(std::min(inc * 2.f, 0.45f));

            float sl = tables.oscillator(phase2, band, false);
            float sr = tables.oscillator(phase3, band, false);
            float body = tables.triangle(phase2, band) * 0.35f;
            float oct = tables.oscillator(phase, oct_band, false) * 0.22f * mod_env;
            float scale = env * gain * 0.30f;
            out = {
                (body + sl * 0.78f + sr * 0.16f + oct) * scale,
                (body + sr * 0.78f + sl * 0.16f - oct * 0.45f) * scale,
            };

            phase += std::min(inc * 2.0f, 0.45f);
            if (phase >= 1.f) phase -= 1.f;
            phase2 += std::min(inc * 0.9942f, 0.45f);
            if (phase2 >= 1.f) phase2 -= 1.f;
            phase3 += std::min(inc * 1.0058f, 0.45f);
            if (phase3 >= 1.f) phase3 -= 1.f;
        } else if (preset_id == 14) {
            // Organic Acoustic Bell-Pluck (Marimba / Nylon Guitar hybrid): woody mallet transient + warm body.
            float inc = std::min(increment * bend_ratio, 0.45f);
            mod_env *= 0.99912f;
            filt_env *= filt_decay;

            float ping = tables.oscillator(phase3, band, true) * (2.2f * mod_env * mod_env);
            float p_mod = phase + ping * 0.22f;
            p_mod -= std::floor(p_mod);

            float body = tables.triangle(p_mod, band) * 0.76f + tables.oscillator(phase, band, true) * 0.24f;
            float overtone = tables.oscillator(phase2, band, true) * 0.34f * mod_env;
            float scale = env * gain * 0.32f;
            out = {(body + overtone) * scale, (body - overtone * 0.75f) * scale};

            phase += inc;
            if (phase >= 1.f) phase -= 1.f;
            phase2 += std::min(inc * 2.0f, 0.45f);
            if (phase2 >= 1.f) phase2 -= 1.f;
            phase3 += std::min(inc * 3.98f, 0.45f);
            if (phase3 >= 1.f) phase3 -= 1.f;
        } else {
            float value = tables.oscillator(phase, band, is_sine) * env * gain * 0.18f;
            out = {value, value};
            phase += std::min(increment * bend_ratio, 0.45f);
            if (phase >= 1.f) phase -= 1.f;
        }

        if (use_vfilt) {
            // Update per-voice resonant filter coefficients every 16 samples (~0.33 ms).
            if ((age & 15u) == 0u) {
                float c = std::clamp(filt_base + filt_depth * filt_env, 120.f, 13500.f);
                vfilt_l.coefficients(c, filt_res);
                vfilt_r.coefficients(c, filt_res);
            }
            out.l = vfilt_l.next(out.l).low;
            out.r = vfilt_r.next(out.r).low;
        }

        ++age;
        if (age == max_age) {
            envelope.off();
        }
        if (!envelope.active()) {
            active = false;
        }
        return out;
    }
};

class VoicePool {
    std::array<Voice, voices_per_limb> voices_{};

public:
    void note_on(
        uint8_t n,
        float value,
        unsigned preset,
        const Wavetables& t,
        SampleView sample,
        bool is_drum_kit = false
    ) noexcept {
        // Prefer: (1) free voice, (2) oldest releasing voice (quietest -> zero click), (3) oldest active voice.
        Voice* selected = nullptr;
        Voice* oldest_rel = nullptr;
        Voice* oldest_any = &voices_[0];

        for (auto& v : voices_) {
            if (!v.active) {
                selected = &v;
                break;
            }
            if (v.envelope.releasing() && (!oldest_rel || v.age > oldest_rel->age)) {
                oldest_rel = &v;
            }
            if (v.age > oldest_any->age) {
                oldest_any = &v;
            }
        }

        if (!selected) {
            selected = oldest_rel ? oldest_rel : oldest_any;
        }
        selected->start(n, value, preset, t, sample, is_drum_kit);
    }

    void note_off(uint8_t n) noexcept {
        for (auto& v : voices_) {
            if (v.active && v.note == n) {
                v.off();
            }
        }
    }

    void silence() noexcept {
        for (auto& v : voices_) {
            v.active = false;
        }
    }

    Stereo next(const Wavetables& tables, float bend_ratio) noexcept {
        Stereo sum{};
        for (auto& v : voices_) {
            auto x = v.next(tables, bend_ratio);
            sum.l += x.l;
            sum.r += x.r;
        }
        return sum;
    }
};

// Classic Juno-style Stereo Ensemble Chorus (fixed 1024-sample buffers, zero heap allocs, zero dry latency).
class StereoChorus {
    static constexpr uint32_t buf_size = 1024;
    std::array<float, buf_size> buf_l_{};
    std::array<float, buf_size> buf_r_{};
    uint32_t write_ = 0;
    float lfo_ = 0.f;

public:
    Stereo next(Stereo in, float mix = 0.26f) noexcept {
        buf_l_[write_] = in.l;
        buf_r_[write_] = in.r;

        lfo_ += 2.f * float(std::numbers::pi) * 0.42f / float(sample_rate);
        if (lfo_ >= 2.f * float(std::numbers::pi)) {
            lfo_ -= 2.f * float(std::numbers::pi);
        }

        float d_l = 240.f + 110.f * std::sin(lfo_);  // 5 ms +/- 2.3 ms quadrature modulation
        float d_r = 240.f + 110.f * std::cos(lfo_);

        auto tap = [&](const std::array<float, buf_size>& buf, float delay) {
            float pos = float(write_) + float(buf_size) - delay;
            uint32_t i0 = uint32_t(pos) & (buf_size - 1);
            uint32_t i1 = (i0 + 1) & (buf_size - 1);
            float frac = pos - std::floor(pos);
            return buf[i0] + (buf[i1] - buf[i0]) * frac;
        };

        float wl = tap(buf_l_, d_l);
        float wr = tap(buf_r_, d_r);
        write_ = (write_ + 1) & (buf_size - 1);

        return {
            in.l * (1.f - mix * 0.35f) + wl * mix,
            in.r * (1.f - mix * 0.35f) + wr * mix,
        };
    }
};

// Zero-latency Presence Bell EQ using CytomicSvf bandpass (brings musical mids to commercial clarity).
class PresenceEq {
    CytomicSvf fl_{};
    CytomicSvf fr_{};
    float boost_ = 0.35f;

public:
    explicit PresenceEq(float hz = 1450.f, float q = 0.85f, float boost = 0.38f)
        : boost_(boost) {
        fl_.coefficients(hz, q);
        fr_.coefficients(hz, q);
    }

    Stereo next(Stereo in) noexcept {
        return {
            in.l + fl_.next(in.l).band * boost_,
            in.r + fr_.next(in.r).band * boost_,
        };
    }
};

// Zero-latency Mono-Sub Crossover (<120 Hz tight mono + gentle stereo width above 120 Hz).
class MonoSubCrossover {
    float lp_l_ = 0.f;
    float lp_r_ = 0.f;
    float alpha_ = 0.f;

public:
    explicit MonoSubCrossover(float hz = 120.f) {
        alpha_ = 1.f - std::exp(-2.f * float(std::numbers::pi) * hz / float(sample_rate));
    }

    Stereo next(Stereo in, float high_width = 1.14f) noexcept {
        lp_l_ += alpha_ * (in.l - lp_l_);
        lp_r_ += alpha_ * (in.r - lp_r_);
        float sub_mono = 0.5f * (lp_l_ + lp_r_);

        float hi_l = in.l - lp_l_;
        float hi_r = in.r - lp_r_;
        float mid = 0.5f * (hi_l + hi_r);
        float side = 0.5f * (hi_l - hi_r) * high_width;
        return {sub_mono + mid + side, sub_mono + mid - side};
    }
};

// Feed-forward SSL-style Master Bus Glue Compressor (zero heap allocs, zero latency).
class MasterGlueCompressor {
    float env_ = 0.f;
    float att_ = 0.f;
    float rel_ = 0.f;

public:
    MasterGlueCompressor() {
        att_ = 1.f - std::exp(-1.f / (sample_rate * 0.012f));  // 12 ms attack preserves transients
        rel_ = 1.f - std::exp(-1.f / (sample_rate * 0.110f));  // 110 ms musical release
    }

    Stereo next(Stereo in) noexcept {
        float peak = std::max(std::abs(in.l), std::abs(in.r));
        float coeff = (peak > env_) ? att_ : rel_;
        env_ += coeff * (peak - env_);

        constexpr float thresh = 0.28f;  // ~ -11 dBFS
        float gr = 1.f;
        if (env_ > thresh) {
            // Smooth 2.4:1 compression ratio above threshold.
            gr = std::pow(env_ / thresh, -0.58f);
        }

        constexpr float makeup = 1.22f;
        return {in.l * gr * makeup, in.r * gr * makeup};
    }
};

// 4-channel Hadamard FDN Stereo Reverb with 1-pole low-pass damping (zero heap allocs).
class Reverb {
    std::array<float, 2069> d0_{};
    std::array<float, 2741> d1_{};
    std::array<float, 3511> d2_{};
    std::array<float, 4373> d3_{};
    std::array<float, 4> lp_{};
    unsigned i0_ = 0;
    unsigned i1_ = 0;
    unsigned i2_ = 0;
    unsigned i3_ = 0;

public:
    Stereo next(Stereo input) noexcept {
        float y0 = d0_[i0_];
        float y1 = d1_[i1_];
        float y2 = d2_[i2_];
        float y3 = d3_[i3_];

        // Silky 1-pole lowpass damping inside feedback loop (warm studio hall, zero high-frequency hiss).
        constexpr float damp = 0.22f;
        constexpr float fb = 0.72f;
        lp_[0] += damp * (y0 - lp_[0]);
        lp_[1] += damp * (y1 - lp_[1]);
        lp_[2] += damp * (y2 - lp_[2]);
        lp_[3] += damp * (y3 - lp_[3]);

        float a = lp_[0];
        float b = lp_[1];
        float c = lp_[2];
        float d = lp_[3];

        // Normalized 4x4 Hadamard orthogonal mixing matrix (factor 0.5).
        float f0 = 0.5f * (a + b + c + d) * fb;
        float f1 = 0.5f * (a - b + c - d) * fb;
        float f2 = 0.5f * (a + b - c - d) * fb;
        float f3 = 0.5f * (a - b - c + d) * fb;

        d0_[i0_] = input.l + f0;
        d1_[i1_] = input.r + f1;
        d2_[i2_] = input.l * 0.7f + input.r * 0.3f + f2;
        d3_[i3_] = input.r * 0.7f + input.l * 0.3f + f3;

        if (++i0_ == d0_.size()) i0_ = 0;
        if (++i1_ == d1_.size()) i1_ = 0;
        if (++i2_ == d2_.size()) i2_ = 0;
        if (++i3_ == d3_.size()) i3_ = 0;

        return {0.5f * (y0 + y2), 0.5f * (y1 + y3)};
    }
};

// Tempo-synced Stereo Ping-Pong Delay (fixed 1-second stereo buffer, zero heap allocs).
class StereoDelay {
    static constexpr uint32_t max_delay = sample_rate;
    std::array<float, max_delay> left_{};
    std::array<float, max_delay> right_{};
    uint32_t write_ = 0;
    float lp_l_ = 0.f;
    float lp_r_ = 0.f;

public:
    Stereo next(Stereo input, float bpm) noexcept {
        // Dotted-eighth note delay (3/16 of a bar = 0.75 * 60 / bpm seconds).
        float seconds = std::clamp(45.f / std::clamp(bpm, 60.f, 200.f), 0.08f, 0.95f);
        uint32_t delay_samples = std::clamp(uint32_t(seconds * sample_rate), 1u, max_delay - 1);
        uint32_t read = (write_ + max_delay - delay_samples) % max_delay;

        float dl = left_[read];
        float dr = right_[read];

        constexpr float damp = 0.30f;
        constexpr float fb = 0.36f;
        lp_l_ += damp * (dl - lp_l_);
        lp_r_ += damp * (dr - lp_r_);

        left_[write_] = input.l + lp_r_ * fb;
        right_[write_] = input.r + lp_l_ * fb;
        if (++write_ == max_delay) {
            write_ = 0;
        }
        return {dl, dr};
    }
};

inline float soft_clip(float x) noexcept {
    if (!finite(x)) {
        return 0.f;
    }
    x = std::clamp(x, -3.f, 3.f);
    float square = x * x;
    return std::clamp(x * (27.f + square) / (27.f + 9.f * square), -1.f, 1.f);
}

inline void limit_stereo(float* samples, uint32_t frames) noexcept {
    unsigned n = 0;
#if defined(__aarch64__)
    // Four independent samples = two stereo frames per NEON iteration.
    const auto lo = vdupq_n_f32(-3.f);
    const auto hi = vdupq_n_f32(3.f);
    const auto c27 = vdupq_n_f32(27.f);
    const auto c9 = vdupq_n_f32(9.f);

    for (; n + 4 <= frames * 2; n += 4) {
        auto x = vld1q_f32(samples + n);
        auto bits = vreinterpretq_u32_f32(x);
        auto normal = vcltq_u32(vandq_u32(bits, vdupq_n_u32(0x7f800000)), vdupq_n_u32(0x7f800000));
        x = vbslq_f32(normal, x, vdupq_n_f32(0.f));
        x = vminq_f32(hi, vmaxq_f32(lo, x));
        auto square = vmulq_f32(x, x);
        auto y = vdivq_f32(vmulq_f32(x, vaddq_f32(c27, square)), vaddq_f32(c27, vmulq_f32(c9, square)));
        vst1q_f32(samples + n, vminq_f32(vdupq_n_f32(1.f), vmaxq_f32(vdupq_n_f32(-1.f), y)));
    }
#endif
    for (; n < frames * 2; ++n) {
        samples[n] = soft_clip(samples[n]);
    }
}

}  // namespace cv_daw
