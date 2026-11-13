#include "moss/settings.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "moss/json.h"

namespace moss {
namespace {

namespace fs = std::filesystem;

std::string settings_path(const std::string &data_dir) { return data_dir + "/settings.json"; }

int clamp_threads(int value) {
    if (value < 0) return 0;
    if (value > 64) return 64;
    return value;
}

}  // namespace

bool load_settings(const std::string &data_dir, EngineSettings *out) {
    if (!out) return false;
    *out = EngineSettings{};
    if (data_dir.empty()) return true;
    std::ifstream in(settings_path(data_dir), std::ios::binary);
    if (!in) return true;  // 没有就是默认值
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    if (text.empty()) return true;

    std::string err;
    JsonValue root;
    if (!json_parse(text, &root, &err) || !root.is_object()) return true;  // 坏文件当默认
    if (const JsonValue *v = root.get("llm_threads")) out->llm_threads = clamp_threads(v->as_int(0));
    if (const JsonValue *v = root.get("decode_threads")) out->decode_threads = clamp_threads(v->as_int(0));
    return true;
}

bool save_settings(const std::string &data_dir, const EngineSettings &in, std::string *err) {
    if (data_dir.empty()) {
        if (err) *err = "没有配置用户数据目录";
        return false;
    }
    std::error_code ec;
    fs::create_directories(data_dir, ec);
    const std::string path = settings_path(data_dir);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (err) *err = "写不了 " + path;
        return false;
    }
    out << "{\n"
        << "  \"llm_threads\": " << clamp_threads(in.llm_threads) << ",\n"
        << "  \"decode_threads\": " << clamp_threads(in.decode_threads) << "\n"
        << "}\n";
    if (!out) {
        if (err) *err = "写不了 " + path;
        return false;
    }
    return true;
}

}  // namespace moss
