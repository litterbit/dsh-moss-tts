#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moss {

// 音色档案（.mossclone）：参考音频编码出来的 16 路离散码，frame-major，几 KB，可以分享给别人。
struct VoiceProfile {
    std::string name;
    int frames = 0;
    int sample_rate = 48000;
    std::vector<int16_t> codes;  // [frames][16]，取值 0..1023
};

bool profile_save(const std::string &path, const VoiceProfile &p, std::string *err);
bool profile_load(const std::string &path, VoiceProfile *p, std::string *err);

}  // namespace moss
