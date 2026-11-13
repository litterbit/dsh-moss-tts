// 用户可调的引擎设置（目前只有两套线程数）。
//
// 落在 <data>/settings.json，启动时读、网页上写；改完要重启进程才生效，
// 因为线程池是随模型加载建的，中途换线程数没有安全的重建路径。
// 优先级：命令行参数 > settings.json > 自动（按物理核，默认 2+2）。
#pragma once

#include <string>

namespace moss {

struct EngineSettings {
    int llm_threads = 0;     // 0 = 自动
    int decode_threads = 0;  // 0 = 自动
};

// 文件不存在或内容坏了都返回 true（按默认值走），只有读不动才返回 false。
bool load_settings(const std::string &data_dir, EngineSettings *out);
bool save_settings(const std::string &data_dir, const EngineSettings &in, std::string *err);

}  // namespace moss
