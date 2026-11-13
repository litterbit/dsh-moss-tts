// 音色的显示名与默认音色。
//
// 音色 id 是磁盘上的文件名（也是 API 里的 `voice` 字段），必须保持 ASCII；
// 页面要显示的是中文名，所以名字单独一张表，只影响显示，不影响任何推理路径。
// 默认音色优先 `manqu`（满满），没装这个音色时退回目录里第一个内置音色，
// 这样只放了权重、或用户把 manqu 删了的环境也不会一开口就报"找不到音色"。
#pragma once

#include <string>

namespace moss {

// 出厂默认音色 id。
inline constexpr const char *kDefaultVoiceId = "manqu";

// 内置音色的中文显示名；未知 id 返回 nullptr（调用侧自己退回 id）。
const char *builtin_voice_label(const std::string &id);

// 目录里按 id 排序的第一个 .mossclone（没有返回空串）。
std::string first_builtin_voice(const std::string &voices_dir);

// 实际使用的默认音色：目录里有 manqu 就用 manqu，否则第一个内置音色，再否则 kDefaultVoiceId。
std::string pick_default_voice(const std::string &voices_dir);

}  // namespace moss
