#pragma once
#include "dsp_primitives.hpp"
#include <fstream>
#include <stdexcept>
namespace cv_daw {
// Called ONLY by offline main loop, never from the live callback.
class WavWriter {
 std::ofstream file_;uint32_t bytes_=0;
 void u16(uint16_t value){char b[2]{char(value),char(value>>8)};file_.write(b,2);}
 void u32(uint32_t value){char b[4]{char(value),char(value>>8),char(value>>16),char(value>>24)};file_.write(b,4);}
 void header(){file_.seekp(0);file_.write("RIFF",4);u32(36+bytes_);file_.write("WAVEfmt ",8);u32(16);u16(1);u16(2);u32(sample_rate);u32(sample_rate*4);u16(4);u16(16);file_.write("data",4);u32(bytes_);}
public:
 explicit WavWriter(const char* path):file_(path,std::ios::binary|std::ios::trunc){if(!file_)throw std::runtime_error("No se puede crear WAV");header();}
 void write(const float* samples,unsigned frames){for(unsigned i=0;i<frames*2;++i)u16(uint16_t(int16_t(std::clamp(samples[i],-1.f,1.f)*32767)));bytes_+=frames*4;if(!file_)throw std::runtime_error("Error escribiendo WAV");}
 ~WavWriter(){header();}
};
}
