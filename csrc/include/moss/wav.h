#pragma once

#include <string>
#include <vector>

namespace moss {

// 读 WAV：PCM16 / PCM24 / PCM32 / IEEE float32（含 WAVE_FORMAT_EXTENSIBLE）。
// 输出交织 float32，范围 [-1, 1]。
bool wav_read(const std::string &path, std::vector<float> *interleaved, int *sample_rate, int *channels,
              std::string *err);

// 写 16-bit PCM WAV（交织）
bool wav_write16(const std::string &path, const std::vector<float> &interleaved, int sample_rate, int channels,
                 std::string *err);

}  // namespace moss
