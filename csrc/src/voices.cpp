#include "moss/voices.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <vector>

namespace moss {
namespace {

namespace fs = std::filesystem;

struct BuiltinName {
    const char *id;
    const char *label;
};

// 内置音色的中文名。顺序无关，只是方便对照 models/voices/ 里有哪些。
// 用户自己克隆的音色不走这张表（它们的 id 就是用户起的名字）。
const BuiltinName kBuiltinNames[] = {
    {"manqu", "满满"},   {"Adam", "亚当"},     {"Anon", "无名"},    {"Arisa", "有纱"},
    {"Ava", "艾娃"},     {"Bella", "贝拉"},    {"Junhao", "俊豪"},  {"Lingyu", "灵语"},
    {"Mei", "梅"},       {"Mortis", "莫提斯"}, {"Nathan", "内森"},  {"Saki", "沙希"},
    {"Soyo", "素世"},    {"Trump", "特朗普"},  {"Umiri", "海铃"},   {"Weiguo", "卫国"},
    {"Xiaoyu", "小雨"},  {"Yuewen", "月雯"},   {"Zhiming", "志明"},
};

}  // namespace

const char *builtin_voice_label(const std::string &id) {
    for (const BuiltinName &entry : kBuiltinNames) {
        if (id == entry.id) return entry.label;
    }
    return nullptr;
}

std::string first_builtin_voice(const std::string &voices_dir) {
    if (voices_dir.empty()) return {};
    std::error_code ec;
    fs::directory_iterator it(voices_dir, ec);
    if (ec) return {};
    std::vector<std::string> ids;
    for (const auto &entry : it) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".mossclone") continue;
        ids.push_back(entry.path().stem().string());
    }
    if (ids.empty()) return {};
    std::sort(ids.begin(), ids.end());
    return ids.front();
}

std::string pick_default_voice(const std::string &voices_dir) {
    if (!voices_dir.empty()) {
        std::error_code ec;
        if (fs::exists(fs::path(voices_dir) / (std::string(kDefaultVoiceId) + ".mossclone"), ec)) {
            return kDefaultVoiceId;
        }
    }
    const std::string first = first_builtin_voice(voices_dir);
    if (!first.empty()) return first;
    return kDefaultVoiceId;
}

}  // namespace moss
