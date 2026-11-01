#pragma once

#include <vector>

namespace moss {

// 任意采样率 -> 目标采样率（Kaiser 窗 sinc，多通道交织）。
// 参考音频要进 48 kHz 的音频 tokenizer，44.1 kHz 的克隆素材必须先重采样。
std::vector<float> resample_interleaved(const float *in, int frames, int channels, int in_rate, int out_rate);

}  // namespace moss
