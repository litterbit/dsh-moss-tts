#include "moss/wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

namespace moss {
namespace {

uint32_t le32(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t le16(const uint8_t *p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

// 压缩格式的 WAV（MP3-in-WAV 就是我们踩过的坑）bits=0、block_align=1，
// 单看块对齐会报出"WAV 块对齐不符合预期"这种没法定位的消息，所以先按格式给个准确的。
const char *fmt_name(int tag) {
    switch (tag) {
        case 0x0001: return "PCM";
        case 0x0003: return "IEEE float";
        case 0x0002: return "MS ADPCM";
        case 0x0006: return "A-law";
        case 0x0007: return "mu-law";
        case 0x0011: return "IMA ADPCM";
        case 0x0050: return "MPEG";
        case 0x0055: return "MP3";
        case 0x00FF: return "AAC";
        case 0xFFFE: return "WAVE_FORMAT_EXTENSIBLE";
        default: return "未知格式";
    }
}

std::string hex_tag(int tag) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%04X", static_cast<unsigned>(tag));
    return buf;
}

}  // namespace

bool wav_read(const std::string &path, std::vector<float> *interleaved, int *sample_rate, int *channels,
              std::string *err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (err) *err = "打不开音频: " + path;
        return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (buf.size() < 44 || std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        if (err) *err = "不是 RIFF/WAVE: " + path;
        return false;
    }

    int fmt_tag = 0, ch = 0, bits = 0, sr = 0, block_align = 0;
    const uint8_t *data = nullptr;
    size_t data_bytes = 0;
    size_t pos = 12;
    while (pos + 8 <= buf.size()) {
        const uint8_t *id = buf.data() + pos;
        const uint32_t len = le32(buf.data() + pos + 4);
        const size_t body = pos + 8;
        if (body + len > buf.size()) break;
        if (std::memcmp(id, "fmt ", 4) == 0 && len >= 16) {
            fmt_tag = le16(buf.data() + body);
            ch = le16(buf.data() + body + 2);
            sr = static_cast<int>(le32(buf.data() + body + 4));
            block_align = le16(buf.data() + body + 12);
            bits = le16(buf.data() + body + 14);
            if (fmt_tag == 0xFFFE && len >= 40) {
                // WAVE_FORMAT_EXTENSIBLE：真正的格式在 SubFormat GUID 的前两字节
                fmt_tag = le16(buf.data() + body + 24);
            }
        } else if (std::memcmp(id, "data", 4) == 0) {
            data = buf.data() + body;
            data_bytes = len;
        }
        pos = body + len + (len & 1);
    }
    if (!data || ch <= 0 || sr <= 0) {
        if (err) *err = "WAV 缺少 fmt/data 块: " + path;
        return false;
    }
    if (fmt_tag != 1 && fmt_tag != 3) {
        if (err) {
            *err = std::string("参考音频是 ") + fmt_name(fmt_tag) + " 编码（" + hex_tag(fmt_tag) +
                   "），不是 PCM；请在网页里直接拖入音频（浏览器会解码），或先把文件转成 PCM WAV";
        }
        return false;
    }
    const int bytes_per_sample = bits / 8;
    if (bytes_per_sample <= 0) {
        if (err) *err = "WAV 位深不合法（" + std::to_string(bits) + " 位）";
        return false;
    }
    if (block_align != bytes_per_sample * ch) {
        if (err) {
            *err = "WAV 块对齐 " + std::to_string(block_align) + " 与 " + std::to_string(ch) + " 声道 × " +
                   std::to_string(bits) + " 位（" + std::to_string(bytes_per_sample * ch) + "）不符";
        }
        return false;
    }
    const size_t frames = data_bytes / static_cast<size_t>(block_align);
    interleaved->resize(frames * static_cast<size_t>(ch));

    for (size_t i = 0; i < frames * static_cast<size_t>(ch); ++i) {
        const uint8_t *p = data + i * static_cast<size_t>(bytes_per_sample);
        float v = 0.0f;
        if (fmt_tag == 3 && bits == 32) {
            float f;
            std::memcpy(&f, p, 4);
            v = f;
        } else if (bits == 16) {
            v = static_cast<float>(static_cast<int16_t>(le16(p))) / 32768.0f;
        } else if (bits == 24) {
            int32_t s = static_cast<int32_t>(p[0] | (p[1] << 8) | (p[2] << 16));
            if (s & 0x800000) s |= ~0xFFFFFF;  // 符号扩展
            v = static_cast<float>(s) / 8388608.0f;
        } else if (bits == 32) {
            v = static_cast<float>(static_cast<int32_t>(le32(p))) / 2147483648.0f;
        } else {
            if (err) *err = "不支持的位深: " + std::to_string(bits);
            return false;
        }
        (*interleaved)[i] = v;
    }
    *sample_rate = sr;
    *channels = ch;
    return true;
}

bool wav_write16(const std::string &path, const std::vector<float> &interleaved, int sample_rate, int channels,
                 std::string *err) {
    std::vector<int16_t> pcm(interleaved.size());
    for (size_t i = 0; i < interleaved.size(); ++i) {
        const float v = std::max(-1.0f, std::min(1.0f, interleaved[i]));
        pcm[i] = static_cast<int16_t>(std::lrintf(v * 32767.0f));
    }
    const uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (err) *err = "写不了 WAV: " + path;
        return false;
    }
    auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char *>(&v), 4); };
    auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char *>(&v), 2); };
    out.write("RIFF", 4);
    u32(36 + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    u32(16);
    u16(1);
    u16(static_cast<uint16_t>(channels));
    u32(static_cast<uint32_t>(sample_rate));
    u32(static_cast<uint32_t>(sample_rate * channels * 2));
    u16(static_cast<uint16_t>(channels * 2));
    u16(16);
    out.write("data", 4);
    u32(data_bytes);
    out.write(reinterpret_cast<const char *>(pcm.data()), data_bytes);
    return static_cast<bool>(out);
}

}  // namespace moss
