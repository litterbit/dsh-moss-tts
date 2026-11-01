#include "moss/profile.h"

#include <cstdio>
#include <cstring>

namespace moss {
namespace {

constexpr char kMagic[4] = {'M', 'S', 'C', 'L'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kNVq = 16;

}  // namespace

bool profile_save(const std::string &path, const VoiceProfile &p, std::string *err) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        if (err) *err = "写不了音色档案: " + path;
        return false;
    }
    const uint32_t frames = static_cast<uint32_t>(p.frames);
    const uint32_t rate = static_cast<uint32_t>(p.sample_rate);
    const uint32_t name_len = static_cast<uint32_t>(p.name.size());
    std::fwrite(kMagic, 1, 4, f);
    std::fwrite(&kVersion, 4, 1, f);
    std::fwrite(&kNVq, 4, 1, f);
    std::fwrite(&frames, 4, 1, f);
    std::fwrite(&rate, 4, 1, f);
    std::fwrite(&name_len, 4, 1, f);
    if (name_len) std::fwrite(p.name.data(), 1, name_len, f);
    std::fwrite(p.codes.data(), sizeof(int16_t), p.codes.size(), f);
    std::fclose(f);
    return true;
}

bool profile_load(const std::string &path, VoiceProfile *p, std::string *err) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "打不开音色档案: " + path;
        return false;
    }
    char magic[4];
    uint32_t version = 0, nvq = 0, frames = 0, rate = 0, name_len = 0;
    const bool ok = std::fread(magic, 1, 4, f) == 4 && std::fread(&version, 4, 1, f) == 1 &&
                    std::fread(&nvq, 4, 1, f) == 1 && std::fread(&frames, 4, 1, f) == 1 &&
                    std::fread(&rate, 4, 1, f) == 1 && std::fread(&name_len, 4, 1, f) == 1;
    if (!ok || std::memcmp(magic, kMagic, 4) != 0 || version != kVersion || nvq != kNVq || frames == 0) {
        std::fclose(f);
        if (err) *err = "音色档案格式不对: " + path;
        return false;
    }
    p->name.assign(name_len, '\0');
    if (name_len && std::fread(p->name.data(), 1, name_len, f) != name_len) {
        std::fclose(f);
        if (err) *err = "音色档案名字被截断: " + path;
        return false;
    }
    p->frames = static_cast<int>(frames);
    p->sample_rate = static_cast<int>(rate);
    p->codes.resize(static_cast<size_t>(frames) * kNVq);
    if (std::fread(p->codes.data(), sizeof(int16_t), p->codes.size(), f) != p->codes.size()) {
        std::fclose(f);
        if (err) *err = "音色档案被截断: " + path;
        return false;
    }
    std::fclose(f);
    return true;
}

}  // namespace moss
