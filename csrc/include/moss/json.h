// 最小 JSON 解析（零依赖）。只做本服务需要的事：解析请求体与 manifest。
//
// 约束：深度 ≤ 8、单个字符串 ≤ 64 KB、禁止 NaN/Inf、拒绝尾随垃圾。
// 不接受重复键的后写覆盖语义（后写覆盖，与 JSON.parse 一致）。
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace moss {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;  // String 的值
    std::vector<JsonValue> items;
    std::vector<std::pair<std::string, JsonValue>> fields;

    const JsonValue *get(const std::string &key) const;
    bool as_bool(bool def = false) const;
    double as_number(double def = 0.0) const;
    int as_int(int def = 0) const;
    std::string as_string(const std::string &def = std::string()) const;
    bool is_object() const { return type == Type::Object; }
};

bool json_parse(const std::string &text, JsonValue *out, std::string *err);

// 输出用：把字符串转义成 JSON 字面量（含 U+0000~U+001F 与引号/反斜杠）
std::string json_escape(const std::string &s);

}  // namespace moss
