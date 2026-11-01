#pragma once

#include <cstdint>
#include <vector>

#include "moss/llm.h"

namespace moss {

// 与上游 browser_poc_manifest.json 一致的特殊 token
constexpr int kAudioStart = 6;
constexpr int kAudioEnd = 7;
constexpr int kUserSlot = 8;    // 参考音频行的 text 位
constexpr int kAssistSlot = 9;  // 生成音频帧的 text 位

// 音色前缀（与正文无关）：12 模板 + audio_start + N 帧音频 + audio_end + 58 模板
int voice_prefix_rows(int voice_frames);

// 参考音频行：每帧一行 [kUserSlot, code0..code15]，codes 为 frame-major
std::vector<Row> build_voice_prefix(const int16_t *codes, int frames);

// 正文段：正文 token 行 + assistant 前缀 + audio_start
std::vector<Row> build_text_suffix(const std::vector<int> &text_ids);

// 文本 token 行：text 位放 token，其余填 audio_pad
Row make_text_row(int token);

}  // namespace moss
