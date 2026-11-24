#pragma once

#include "dsp_primitives.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace cv_daw {

// Called ONLY by offline main loop or background LiveTap thread, never from the live audio callback.
class WavWriter {
    std::ofstream file_;
    uint32_t bytes_ = 0;

    void write_u16(uint16_t value) {
        char bytes[2]{char(value), char(value >> 8)};
        file_.write(bytes, 2);
    }

    void write_u32(uint32_t value) {
        char bytes[4]{
            char(value),
            char(value >> 8),
            char(value >> 16),
            char(value >> 24),
        };
        file_.write(bytes, 4);
    }

    void write_header() {
        file_.seekp(0);
        file_.write("RIFF", 4);
        write_u32(36 + bytes_);
        file_.write("WAVEfmt ", 8);
        write_u32(16);
        write_u16(1);               // PCM
        write_u16(2);               // Stereo
        write_u32(sample_rate);
        write_u32(sample_rate * 4); // Byte rate
        write_u16(4);               // Block align
        write_u16(16);              // Bits per sample
        file_.write("data", 4);
        write_u32(bytes_);
    }

public:
    explicit WavWriter(const char* path)
        : file_(path, std::ios::binary | std::ios::trunc) {
        if (!file_) {
            throw std::runtime_error("No se puede crear WAV");
        }
        write_header();
    }

    void write(const float* samples, unsigned frames) {
        for (unsigned i = 0; i < frames * 2; ++i) {
            float clamped = std::clamp(samples[i], -1.f, 1.f);
            write_u16(uint16_t(int16_t(clamped * 32767)));
        }
        bytes_ += frames * 4;
        if (!file_) {
            throw std::runtime_error("Error escribiendo WAV");
        }
    }

    ~WavWriter() {
        write_header();
    }
};

}  // namespace cv_daw
