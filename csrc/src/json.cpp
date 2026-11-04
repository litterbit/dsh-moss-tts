#include "moss/json.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace moss {
namespace {

constexpr int kMaxDepth = 8;
constexpr size_t kMaxString = 64 * 1024;
constexpr size_t kMaxNodes = 4096;

struct Parser {
    const std::string &s;
    size_t i = 0;
    size_t nodes = 0;
    std::string err;

    explicit Parser(const std::string &text) : s(text) {}

    bool fail(const char *msg) {
        if (err.empty()) err = msg;
        return false;
    }

    void skip_ws() {
        while (i < s.size()) {
            const char c = s[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++i;
            } else {
                break;
            }
        }
    }

    static void append_utf8(std::string *out, unsigned cp) {
        if (cp < 0x80) {
            out->push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    bool parse_hex4(unsigned *out) {
        if (i + 4 > s.size()) return fail("\\u 转义不完整");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i + static_cast<size_t>(k)];
            v <<= 4;
            if (c >= '0' && c <= '9') {
                v |= static_cast<unsigned>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                v |= static_cast<unsigned>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                v |= static_cast<unsigned>(c - 'A' + 10);
            } else {
                return fail("\\u 转义含非法字符");
            }
        }
        i += 4;
        *out = v;
        return true;
    }

    bool parse_string(std::string *out) {
        if (i >= s.size() || s[i] != '"') return fail("期望字符串");
        ++i;
        while (i < s.size()) {
            const unsigned char c = static_cast<unsigned char>(s[i]);
            if (c == '"') {
                ++i;
                return true;
            }
            if (c < 0x20) return fail("字符串里有未转义的控制字符");
            if (c != '\\') {
                out->push_back(static_cast<char>(c));
                ++i;
                if (out->size() > kMaxString) return fail("字符串过长");
                continue;
            }
            ++i;
            if (i >= s.size()) return fail("转义不完整");
            const char e = s[i++];
            switch (e) {
                case '"': out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/': out->push_back('/'); break;
                case 'b': out->push_back('\b'); break;
                case 'f': out->push_back('\f'); break;
                case 'n': out->push_back('\n'); break;
                case 'r': out->push_back('\r'); break;
                case 't': out->push_back('\t'); break;
                case 'u': {
                    unsigned cp = 0;
                    if (!parse_hex4(&cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (i + 1 >= s.size() || s[i] != '\\' || s[i + 1] != 'u') {
                            return fail("高位代理后面缺少低位代理");
                        }
                        i += 2;
                        unsigned lo = 0;
                        if (!parse_hex4(&lo)) return false;
                        if (lo < 0xDC00 || lo > 0xDFFF) return fail("低位代理非法");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("孤立的低位代理");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: return fail("未知转义字符");
            }
            if (out->size() > kMaxString) return fail("字符串过长");
        }
        return fail("字符串没有结束");
    }

    bool parse_number(double *out) {
        const size_t start = i;
        if (i < s.size() && s[i] == '-') ++i;
        size_t digits = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            ++i;
            ++digits;
        }
        if (digits == 0) return fail("数字格式不对");
        if (i < s.size() && s[i] == '.') {
            ++i;
            size_t frac = 0;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                ++i;
                ++frac;
            }
            if (frac == 0) return fail("小数部分为空");
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            size_t exp = 0;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                ++i;
                ++exp;
            }
            if (exp == 0) return fail("指数部分为空");
        }
        const std::string token = s.substr(start, i - start);
        char *end = nullptr;
        const double v = std::strtod(token.c_str(), &end);
        if (end == nullptr || *end != '\0') return fail("数字格式不对");
        if (!std::isfinite(v)) return fail("数字溢出");
        *out = v;
        return true;
    }

    bool parse_value(JsonValue *out, int depth) {
        if (depth > kMaxDepth) return fail("嵌套过深");
        if (++nodes > kMaxNodes) return fail("结构过大");
        skip_ws();
        if (i >= s.size()) return fail("提前结束");
        const char c = s[i];
        if (c == '{') {
            out->type = JsonValue::Type::Object;
            ++i;
            skip_ws();
            if (i < s.size() && s[i] == '}') {
                ++i;
                return true;
            }
            while (true) {
                skip_ws();
                std::string key;
                if (!parse_string(&key)) return false;
                skip_ws();
                if (i >= s.size() || s[i] != ':') return fail("对象里缺少冒号");
                ++i;
                JsonValue value;
                if (!parse_value(&value, depth + 1)) return false;
                bool replaced = false;
                for (auto &kv : out->fields) {
                    if (kv.first == key) {
                        kv.second = std::move(value);
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) out->fields.emplace_back(std::move(key), std::move(value));
                skip_ws();
                if (i >= s.size()) return fail("对象没有结束");
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                if (s[i] == '}') {
                    ++i;
                    return true;
                }
                return fail("对象里缺少逗号或右花括号");
            }
        }
        if (c == '[') {
            out->type = JsonValue::Type::Array;
            ++i;
            skip_ws();
            if (i < s.size() && s[i] == ']') {
                ++i;
                return true;
            }
            while (true) {
                JsonValue value;
                if (!parse_value(&value, depth + 1)) return false;
                out->items.push_back(std::move(value));
                skip_ws();
                if (i >= s.size()) return fail("数组没有结束");
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                if (s[i] == ']') {
                    ++i;
                    return true;
                }
                return fail("数组里缺少逗号或右方括号");
            }
        }
        if (c == '"') {
            out->type = JsonValue::Type::String;
            return parse_string(&out->text);
        }
        if (s.compare(i, 4, "true") == 0) {
            out->type = JsonValue::Type::Bool;
            out->boolean = true;
            i += 4;
            return true;
        }
        if (s.compare(i, 5, "false") == 0) {
            out->type = JsonValue::Type::Bool;
            out->boolean = false;
            i += 5;
            return true;
        }
        if (s.compare(i, 4, "null") == 0) {
            out->type = JsonValue::Type::Null;
            i += 4;
            return true;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            out->type = JsonValue::Type::Number;
            return parse_number(&out->number);
        }
        return fail("无法识别的值");
    }
};

}  // namespace

const JsonValue *JsonValue::get(const std::string &key) const {
    for (const auto &kv : fields) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

bool JsonValue::as_bool(bool def) const { return type == Type::Bool ? boolean : def; }

double JsonValue::as_number(double def) const { return type == Type::Number ? number : def; }

int JsonValue::as_int(int def) const {
    if (type != Type::Number) return def;
    if (number < -1e9 || number > 1e9) return def;
    return static_cast<int>(number < 0 ? number - 0.5 : number + 0.5);
}

std::string JsonValue::as_string(const std::string &def) const {
    return type == Type::String ? text : def;
}

bool json_parse(const std::string &text, JsonValue *out, std::string *err) {
    Parser parser(text);
    JsonValue value;
    if (!parser.parse_value(&value, 0)) {
        if (err) *err = parser.err.empty() ? "JSON 解析失败" : parser.err;
        return false;
    }
    parser.skip_ws();
    if (parser.i != text.size()) {
        if (err) *err = "JSON 结尾有多余内容";
        return false;
    }
    *out = std::move(value);
    return true;
}

std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char *hex = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(hex[(c >> 4) & 0xF]);
                    out.push_back(hex[c & 0xF]);
                } else {
                    out.push_back(ch);
                }
        }
    }
    return out;
}

}  // namespace moss
