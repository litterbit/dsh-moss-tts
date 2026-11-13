#include "moss/http_server.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "moss/json.h"
#include "moss/runtime.h"
#include "moss/settings.h"
#include "moss/version.h"

namespace fs = std::filesystem;

namespace moss {
namespace {

constexpr size_t kMaxHeaderBytes = 16 * 1024;
constexpr size_t kMaxJsonBody = 256 * 1024;
constexpr size_t kMaxUploadBody = 20u * 1024 * 1024;
constexpr int kRecvTimeoutMs = 10000;
constexpr int kSendTimeoutMs = 30000;
constexpr int kJobTimeoutMs = 600000;

// 项目门户的 origin。默认放行，这样用户双击 exe 之后，Pages 上的页面可以直接连本机引擎。
// 想完全关掉跨源就传 `--allow-origin none`（见 tts_main.cpp）。
constexpr const char *kProjectOrigins[] = {"https://litterbit.github.io"};

struct Request {
    std::string method;
    std::string target;
    std::string path;
    std::string query;
    std::string version;
    std::map<std::string, std::string> headers;  // 名称小写
    std::string body;

    std::string header(const std::string &name) const {
        const auto it = headers.find(name);
        return it == headers.end() ? std::string() : it->second;
    }
};

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string trim(const std::string &s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
    return s.substr(b, e - b);
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string percent_decode(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const int hi = hex_value(s[i + 1]);
            const int lo = hex_value(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>(hi * 16 + lo));
                i += 2;
                continue;
            }
        }
        out.push_back(s[i]);
    }
    return out;
}

std::map<std::string, std::string> parse_query(const std::string &query) {
    std::map<std::string, std::string> out;
    size_t start = 0;
    while (start <= query.size()) {
        const size_t amp = query.find('&', start);
        const std::string pair = query.substr(start, amp == std::string::npos ? std::string::npos : amp - start);
        if (!pair.empty()) {
            const size_t eq = pair.find('=');
            const std::string key = percent_decode(pair.substr(0, eq));
            std::string value = eq == std::string::npos ? std::string() : percent_decode(pair.substr(eq + 1));
            std::replace(value.begin(), value.end(), '+', ' ');
            if (!key.empty()) out[key] = value;
        }
        if (amp == std::string::npos) break;
        start = amp + 1;
    }
    return out;
}

bool send_all(SOCKET sock, const void *data, size_t size) {
    const char *p = static_cast<const char *>(data);
    size_t left = size;
    while (left > 0) {
        const int n = ::send(sock, p, static_cast<int>(std::min<size_t>(left, 1 << 20)), 0);
        if (n <= 0) return false;
        p += n;
        left -= static_cast<size_t>(n);
    }
    return true;
}

const char *status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 429: return "Too Many Requests";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "OK";
    }
}

// 用当前命令行再起一个自己（网页上的"重启"）。
// 新进程固定带 --no-open（重启不该再弹一个浏览器窗口）和 --port-wait（等旧进程松开端口），
// 旧进程发完响应就退出。
bool spawn_restart(std::string *err) {
    wchar_t exe[MAX_PATH] = {};
    if (GetModuleFileNameW(nullptr, exe, MAX_PATH) == 0) {
        if (err) *err = "取不到 exe 路径";
        return false;
    }
    std::wstring cmd = GetCommandLineW();
    // 原始命令行可能就只是 exe 路径（双击来的，"不带参数 = 起服务"）。子进程必须显式带 --serve，
    // 否则 argc>1 时那一层判断不成立，新进程会打印用法然后退出。
    if (cmd.find(L"--serve") == std::wstring::npos) cmd += L" --serve";
    if (cmd.find(L"--no-open") == std::wstring::npos) cmd += L" --no-open";
    if (cmd.find(L"--port-wait") == std::wstring::npos) cmd += L" --port-wait 5000";
    std::vector<wchar_t> line(cmd.begin(), cmd.end());
    line.push_back(L'\0');

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    // 先试着脱离当前 Job 对象：被别的程序（比如编辑器的任务、CI）用 kill-on-close job 拉起时，
    // 子进程会跟着父进程一起被杀，重启就成了"关掉自己"。Job 不允许脱离时退回普通启动。
    DWORD flags = CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB;
    if (!CreateProcessW(exe, line.data(), nullptr, nullptr, FALSE, flags, nullptr, nullptr, &si, &pi)) {
        flags = CREATE_NEW_PROCESS_GROUP;
        if (!CreateProcessW(exe, line.data(), nullptr, nullptr, FALSE, flags, nullptr, nullptr, &si, &pi)) {
            if (err) *err = "启动新进程失败（错误码 " + std::to_string(GetLastError()) + "）";
            return false;
        }
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

struct Response {
    SOCKET sock = INVALID_SOCKET;
    std::string cors_origin;  // 非空则加 ACAO
    bool headers_sent = false;

    bool headers(int status, const std::vector<std::pair<std::string, std::string>> &extra) {
        if (headers_sent) return false;
        headers_sent = true;
        std::ostringstream out;
        out << "HTTP/1.1 " << status << " " << status_text(status) << "\r\n";
        out << "Connection: close\r\n";
        out << "X-Moss-Version: " << MOSS_VERSION << "\r\n";
        if (!cors_origin.empty()) {
            out << "Access-Control-Allow-Origin: " << cors_origin << "\r\n";
            out << "Vary: Origin\r\n";
        }
        for (const auto &kv : extra) out << kv.first << ": " << kv.second << "\r\n";
        out << "\r\n";
        const std::string head = out.str();
        return send_all(sock, head.data(), head.size());
    }

    bool json(int status, const std::string &body) {
        const std::string payload = "{\"ok\":" + std::string(status == 200 ? "true" : "false") + "," + body + "}";
        return raw_json(status, payload);
    }

    bool raw_json(int status, const std::string &payload) {
        if (!headers(status, {{"Content-Type", "application/json; charset=utf-8"},
                              {"Content-Length", std::to_string(payload.size())},
                              {"Cache-Control", "no-store"}})) {
            return false;
        }
        return send_all(sock, payload.data(), payload.size());
    }

    bool error(int status, const std::string &code, const std::string &message) {
        const std::string payload = "{\"ok\":false,\"code\":\"" + json_escape(code) + "\",\"error\":\"" +
                                    json_escape(message) + "\"}";
        return raw_json(status, payload);
    }
};

bool read_request(SOCKET sock, Request *req, size_t max_body, std::string *err) {
    std::string buffer;
    char chunk[8192];
    size_t header_end = std::string::npos;
    while (true) {
        const int n = ::recv(sock, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            *err = "读取请求失败";
            return false;
        }
        buffer.append(chunk, static_cast<size_t>(n));
        header_end = buffer.find("\r\n\r\n");
        if (header_end != std::string::npos) break;
        if (buffer.size() > kMaxHeaderBytes) {
            *err = "请求头过大";
            return false;
        }
    }

    const std::string head = buffer.substr(0, header_end);
    std::istringstream lines(head);
    std::string line;
    if (!std::getline(lines, line)) {
        *err = "请求行缺失";
        return false;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    {
        std::istringstream first(line);
        if (!(first >> req->method >> req->target >> req->version)) {
            *err = "请求行格式不对";
            return false;
        }
    }
    if (req->version != "HTTP/1.1" && req->version != "HTTP/1.0") {
        *err = "只支持 HTTP/1.1";
        return false;
    }
    const size_t q = req->target.find('?');
    req->path = q == std::string::npos ? req->target : req->target.substr(0, q);
    req->query = q == std::string::npos ? std::string() : req->target.substr(q + 1);

    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            *err = "请求头格式不对";
            return false;
        }
        const std::string name = lower(trim(line.substr(0, colon)));
        const std::string value = trim(line.substr(colon + 1));
        if (req->headers.count(name)) {
            // 重复头一律拒绝：Content-Length 冲突与请求走私都从这里来
            *err = "请求头重复：" + name;
            return false;
        }
        req->headers[name] = value;
    }

    if (!req->header("transfer-encoding").empty()) {
        *err = "不支持分块上传";
        return false;
    }
    size_t content_length = 0;
    const std::string cl = req->header("content-length");
    if (!cl.empty()) {
        for (const char c : cl) {
            if (c < '0' || c > '9') {
                *err = "Content-Length 非法";
                return false;
            }
        }
        if (cl.size() > 10) {
            *err = "Content-Length 过大";
            return false;
        }
        content_length = static_cast<size_t>(std::strtoull(cl.c_str(), nullptr, 10));
        if (content_length > max_body) {
            *err = "请求体过大";
            return false;
        }
    }
    if (req->method != "GET" && req->method != "HEAD" && req->method != "OPTIONS" && cl.empty()) {
        // 允许无体的 POST（例如 /jobs/{id}/cancel）：不读 body，长度按 0 处理
        content_length = 0;
    }

    size_t have = buffer.size() - (header_end + 4);
    req->body = buffer.substr(header_end + 4);
    while (have < content_length) {
        const int n = ::recv(sock, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            *err = "请求体读取失败";
            return false;
        }
        req->body.append(chunk, static_cast<size_t>(n));
        have += static_cast<size_t>(n);
        if (req->body.size() > max_body) {
            *err = "请求体过大";
            return false;
        }
    }
    if (req->body.size() > content_length) req->body.resize(content_length);
    return true;
}

std::string mime_for(const std::string &path) {
    const std::string ext = lower(fs::path(path).extension().string());
    if (ext == ".html" || ext == ".htm") return "text/html; charset=utf-8";
    if (ext == ".js" || ext == ".mjs") return "text/javascript; charset=utf-8";
    if (ext == ".css") return "text/css; charset=utf-8";
    if (ext == ".json") return "application/json; charset=utf-8";
    if (ext == ".wav") return "audio/wav";
    if (ext == ".svg") return "image/svg+xml";
    if (ext == ".png") return "image/png";
    if (ext == ".ico") return "image/x-icon";
    if (ext == ".woff2") return "font/woff2";
    return "application/octet-stream";
}

bool send_file(Response &resp, const fs::path &root, const std::string &url_path, const std::string &cors) {
    std::string rel = percent_decode(url_path);
    if (rel.empty() || rel == "/") rel = "/index.html";
    if (rel.find('\\') != std::string::npos || rel.find('\0') != std::string::npos) {
        return resp.error(400, "bad_path", "路径非法");
    }
    std::vector<std::string> parts;
    std::string current;
    for (const char c : rel) {
        if (c == '/') {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) parts.push_back(current);
    for (const auto &p : parts) {
        if (p == ".." || p == ".") return resp.error(400, "bad_path", "路径非法");
    }
    fs::path full = root;
    for (const auto &p : parts) full /= p;

    std::error_code ec;
    const fs::path root_canon = fs::weakly_canonical(root, ec);
    const fs::path full_canon = fs::weakly_canonical(full, ec);
    if (ec) return resp.error(404, "not_found", "文件不存在");
    const std::string root_s = root_canon.string();
    const std::string full_s = full_canon.string();
    if (full_s.size() <= root_s.size() || full_s.compare(0, root_s.size(), root_s) != 0) {
        return resp.error(403, "forbidden", "越界访问");
    }
    if (!fs::is_regular_file(full_canon, ec)) return resp.error(404, "not_found", "文件不存在");

    std::ifstream in(full_canon, std::ios::binary);
    if (!in) return resp.error(404, "not_found", "文件读不了");
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string body = ss.str();

    const std::string mime = mime_for(full_canon.string());
    std::vector<std::pair<std::string, std::string>> extra = {
        {"Content-Type", mime},
        {"Content-Length", std::to_string(body.size())},
        {"Cache-Control", mime.rfind("text/html", 0) == 0 ? "no-cache" : "public, max-age=600"},
        {"X-Content-Type-Options", "nosniff"},
        {"Referrer-Policy", "no-referrer"},
    };
    if (mime.rfind("text/html", 0) == 0) {
        extra.emplace_back("Content-Security-Policy",
                           "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                           "media-src 'self' blob: data:; img-src 'self' data:; connect-src 'self'; "
                           "frame-ancestors 'none'; base-uri 'none'");
    }
    resp.cors_origin = cors;
    if (!resp.headers(200, extra)) return false;
    return send_all(resp.sock, body.data(), body.size());
}

std::string job_json(const JobInfo &info) {
    std::ostringstream out;
    out << "\"id\":\"" << json_escape(info.id) << "\"";
    out << ",\"state\":\"" << job_state_name(info.state) << "\"";
    out << ",\"frames\":" << info.frames;
    out << ",\"audio_ms\":" << static_cast<long long>(info.audio_ms);
    out << ",\"compute_ms\":" << static_cast<long long>(info.compute_ms);
    out << ",\"wall_ms\":" << static_cast<long long>(info.wall_ms);
    out << ",\"first_pcm_ms\":" << static_cast<long long>(info.first_pcm_ms);
    out << ",\"rtf\":" << (info.rtf > 0 ? info.rtf : 0.0);
    out << ",\"text_tokens\":" << info.text_tokens;
    out << ",\"cache_hit\":" << (info.cache_hit ? "true" : "false");
    out << ",\"error\":\"" << json_escape(info.error) << "\"";
    return out.str();
}

std::string voices_json(const Runtime &rt) {
    std::ostringstream out;
    const std::string def = rt.default_voice();
    std::vector<VoiceInfo> all = rt.voices();
    auto one = [&](const VoiceInfo &v) {
        out << "{\"id\":\"" << json_escape(v.id) << "\",\"label\":\"" << json_escape(v.label)
            << "\",\"frames\":" << v.frames << ",\"seconds\":" << v.seconds
            << ",\"default\":" << (v.id == def ? "true" : "false") << "}";
    };
    out << "\"builtin\":[";
    bool first = true;
    for (const auto &v : all) {
        if (v.user) continue;
        if (!first) out << ",";
        first = false;
        one(v);
    }
    out << "],\"user\":[";
    first = true;
    for (const auto &v : all) {
        if (!v.user) continue;
        if (!first) out << ",";
        first = false;
        one(v);
    }
    out << "],\"default\":\"" << json_escape(def) << "\"";
    return out.str();
}

bool parse_job_request(const JsonValue &root, JobRequest *req, std::string *err) {
    if (!root.is_object()) {
        *err = "请求体必须是 JSON 对象";
        return false;
    }
    req->text = root.get("text") ? root.get("text")->as_string() : std::string();
    if (req->text.empty()) {
        *err = "text 不能为空";
        return false;
    }
    if (const JsonValue *v = root.get("voice")) req->voice = v->as_string();
    // 注意：不接受客户端给的文件路径（否则等于开了任意文件读）。只接受音色 id + scope，
    // 路径由服务端在两个音色目录里拼出来。
    req->profile.clear();
    if (const JsonValue *v = root.get("seed")) {
        const double d = v->as_number(0);
        req->seed = d < 0 ? 0 : static_cast<uint64_t>(d);
    }
    if (const JsonValue *v = root.get("greedy")) req->greedy = v->as_bool(false);
    if (const JsonValue *v = root.get("max_frames")) req->max_frames = v->as_int(375);
    if (const JsonValue *v = root.get("chunk_frames")) req->chunk_frames = v->as_int(8);
    if (const JsonValue *v = root.get("min_frames")) req->min_frames = v->as_int(0);
    if (const JsonValue *v = root.get("temperature")) req->temperature = static_cast<float>(v->as_number(0));
    if (const JsonValue *v = root.get("top_k")) req->top_k = v->as_int(0);
    if (const JsonValue *v = root.get("top_p")) req->top_p = static_cast<float>(v->as_number(0));
    if (const JsonValue *v = root.get("repetition_penalty"))
        req->repetition_penalty = static_cast<float>(v->as_number(0));
    if (const JsonValue *v = root.get("pace")) req->pace = v->as_number(0);

    req->max_frames = std::clamp(req->max_frames, 1, 2400);
    req->chunk_frames = std::clamp(req->chunk_frames, 1, 64);
    req->min_frames = std::clamp(req->min_frames, 0, 2400);
    if (req->temperature < 0.0f || req->temperature > 2.0f) req->temperature = 0.0f;
    if (req->top_k != 0) req->top_k = std::clamp(req->top_k, 1, 1024);
    if (!(req->top_p >= 0.0f && req->top_p <= 1.0f)) req->top_p = 0.0f;
    if (req->repetition_penalty < 0.0f || req->repetition_penalty > 2.0f) req->repetition_penalty = 0.0f;
    if (!(req->pace >= 0.0 && req->pace <= 30.0)) req->pace = 0.0;
    return true;
}

bool valid_voice_id(const std::string &id) {
    if (id.empty() || id.size() > 32) return false;
    for (const char c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                        c == '-';
        if (!ok) return false;
    }
    return true;
}

std::string make_request_id() {
    static std::atomic<uint64_t> counter{0};
    const uint64_t n = ++counter;
    const uint64_t t = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%08llx%08llx", static_cast<unsigned long long>(t & 0xFFFFFFFFu),
                  static_cast<unsigned long long>(n & 0xFFFFFFFFu));
    return buf;
}

bool send_chunk_header(SOCKET sock, size_t len) {
    char head[32];
    const int n = std::snprintf(head, sizeof(head), "%llx\r\n", static_cast<unsigned long long>(len));
    return send_all(sock, head, static_cast<size_t>(n));
}

bool send_pcm_chunk(SOCKET sock, const void *data, size_t bytes) {
    if (!send_chunk_header(sock, bytes)) return false;
    if (!send_all(sock, data, bytes)) return false;
    return send_all(sock, "\r\n", 2);
}

bool send_stream_headers(Response *resp, const std::string &request_id) {
    return resp->headers(200, {{"Content-Type", "application/octet-stream"},
                               {"Transfer-Encoding", "chunked"},
                               {"Cache-Control", "no-store"},
                               {"X-Content-Type-Options", "nosniff"},
                               {"X-Moss-Request-Id", request_id},
                               {"X-Moss-Audio-Format", "s16le;rate=48000;channels=2"},
                               {"X-Moss-Api-Version", MOSS_API_VERSION}});
}

std::string build_wav(const std::vector<int16_t> &pcm, int sample_rate, int channels) {
    const uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * channels * 2);
    const uint16_t block_align = static_cast<uint16_t>(channels * 2);
    std::string out;
    out.reserve(44 + data_bytes);
    auto put32 = [&](uint32_t v) {
        out.push_back(static_cast<char>(v & 0xFF));
        out.push_back(static_cast<char>((v >> 8) & 0xFF));
        out.push_back(static_cast<char>((v >> 16) & 0xFF));
        out.push_back(static_cast<char>((v >> 24) & 0xFF));
    };
    auto put16 = [&](uint16_t v) {
        out.push_back(static_cast<char>(v & 0xFF));
        out.push_back(static_cast<char>((v >> 8) & 0xFF));
    };
    out += "RIFF";
    put32(36 + data_bytes);
    out += "WAVEfmt ";
    put32(16);
    put16(1);
    put16(static_cast<uint16_t>(channels));
    put32(static_cast<uint32_t>(sample_rate));
    put32(byte_rate);
    put16(block_align);
    put16(16);
    out += "data";
    put32(data_bytes);
    out.append(reinterpret_cast<const char *>(pcm.data()), data_bytes);
    return out;
}

void open_browser(const std::string &url) {
    const int len = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, nullptr, 0);
    if (len <= 0) return;
    std::wstring wide(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, wide.data(), len);
    ShellExecuteW(nullptr, L"open", wide.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

class Server {
public:
    Server(Runtime &rt, const ServerOptions &opt, int port)
        : rt_(rt), opt_(opt), port_(port), self_origins_{"http://127.0.0.1:" + std::to_string(port),
                                                          "http://localhost:" + std::to_string(port)} {
        // 自己的门户页：允许它直接连本机引擎（"打开网页就能用"的那条路）。
        // 只放行我们这一个 origin，不是 *.github.io —— 每个 GitHub Pages 用户是独立 origin。
        if (opt_.allow_project_origin) {
            for (const char *origin : kProjectOrigins) self_origins_.emplace_back(origin);
        }
        for (const auto &extra : opt_.allow_origins) self_origins_.push_back(extra);
    }

    void handle(SOCKET sock);

private:
    bool host_ok(const Request &req) const;
    std::string allowed_origin(const Request &req) const;
    void dispatch(SOCKET sock, Request &req);
    void handle_options(SOCKET sock, const Request &req);
    void handle_tts(SOCKET sock, const Request &req, bool wav);
    void handle_clone(SOCKET sock, const Request &req);

    Runtime &rt_;
    const ServerOptions opt_;
    int port_;
    std::vector<std::string> self_origins_;
};

bool Server::host_ok(const Request &req) const {
    const std::string host = lower(req.header("host"));
    if (host.empty()) return true;  // 非浏览器客户端（如 curl）可以不带 Host
    const std::string p = std::to_string(port_);
    return host == "127.0.0.1:" + p || host == "localhost:" + p || host == "[::1]:" + p ||
           host == "127.0.0.1" || host == "localhost";
}

std::string Server::allowed_origin(const Request &req) const {
    const std::string origin = lower(req.header("origin"));
    if (origin.empty()) return std::string();
    for (const auto &o : self_origins_) {
        if (lower(o) == origin) return req.header("origin");
    }
    for (const auto &o : opt_.allow_origins) {
        if (lower(o) == origin) return req.header("origin");
    }
    return std::string();
}

void Server::handle(SOCKET sock) {
    DWORD recv_timeout = kRecvTimeoutMs;
    DWORD send_timeout = kSendTimeoutMs;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&recv_timeout), sizeof(recv_timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&send_timeout), sizeof(send_timeout));

    Request req;
    std::string err;
    // 上限先按上传给（克隆要 20 MB），JSON 接口在 dispatch 里再卡 256 KB
    if (!read_request(sock, &req, kMaxUploadBody, &err)) {
        Response resp;
        resp.sock = sock;
        resp.error(400, "bad_request", err);
        closesocket(sock);
        return;
    }
    dispatch(sock, req);
    closesocket(sock);
}

void Server::handle_options(SOCKET sock, const Request &req) {
    Response resp;
    resp.sock = sock;
    const std::string origin = allowed_origin(req);
    if (origin.empty()) {
        resp.error(403, "origin_denied", "该来源不被允许");
        return;
    }
    resp.cors_origin = origin;
    std::vector<std::pair<std::string, std::string>> extra = {
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "content-type, x-moss-request-id"},
        {"Access-Control-Max-Age", "600"},
        {"Content-Length", "0"},
    };
    if (lower(req.header("access-control-request-private-network")) == "true") {
        extra.emplace_back("Access-Control-Allow-Private-Network", "true");
    }
    resp.headers(204, extra);
}

void Server::handle_tts(SOCKET sock, const Request &req, bool want_wav) {
    Response resp;
    resp.sock = sock;
    resp.cors_origin = allowed_origin(req);

    JsonValue root;
    std::string parse_err;
    if (!json_parse(req.body, &root, &parse_err)) {
        resp.error(400, "bad_json", parse_err);
        return;
    }
    JobRequest job;
    if (!parse_job_request(root, &job, &parse_err)) {
        resp.error(400, "bad_request", parse_err);
        return;
    }
    job.id = make_request_id();

    // 音色解析：只接受 id，路径由服务端拼；用户音色和内置音色分目录
    {
        const std::string scope = root.get("scope") ? root.get("scope")->as_string("builtin") : "builtin";
        // 没指定音色 = 出厂默认（manqu/满满），与 CLI、插件走同一套逻辑
        if (job.voice.empty()) job.voice = rt_.default_voice();
        if (!valid_voice_id(job.voice)) {
            resp.error(400, "bad_voice", "音色名不合法");
            return;
        }
        const std::string base = scope == "user" ? rt_.options().user_voices_dir : rt_.options().voices_dir;
        if (base.empty()) {
            resp.error(500, "no_voices_dir", "没有配置音色目录");
            return;
        }
        const std::string path = base + "/" + job.voice + ".mossclone";
        std::error_code ec;
        if (!fs::is_regular_file(path, ec)) {
            resp.error(404, "voice_not_found", "找不到音色：" + job.voice);
            return;
        }
        job.profile = path;
    }

    if (want_wav) {
        std::vector<int16_t> pcm;
        std::string submit_err;
        const bool trim_tail = root.get("trim") && root.get("trim")->as_bool(false);
        auto sink = [&pcm](const int16_t *data, int frames) {
            pcm.insert(pcm.end(), data, data + static_cast<size_t>(frames) * 2);
            return true;
        };
        std::string job_id;
        if (!rt_.submit(job, sink, &job_id, &submit_err)) {
            resp.error(submit_err == "busy" ? 429 : 500, submit_err == "busy" ? "busy" : "submit_failed", submit_err);
            return;
        }
        JobInfo info;
        rt_.wait(job_id, kJobTimeoutMs, &info);
        if (info.state == JobState::Failed) {
            resp.error(500, "synth_failed", info.error);
            return;
        }
        if (trim_tail) trim_tail_silence(&pcm, 48000, 2);
        const std::string wav = build_wav(pcm, 48000, 2);
        std::vector<std::pair<std::string, std::string>> extra = {
            {"Content-Type", "audio/wav"},
            {"Content-Length", std::to_string(wav.size())},
            {"Cache-Control", "no-store"},
            {"X-Moss-Request-Id", job_id},
            {"X-Moss-Audio-Format", "s16le;rate=48000;channels=2"},
            {"Content-Disposition", "attachment; filename=\"moss-tts.wav\""},
        };
        if (!resp.headers(200, extra)) return;
        send_all(sock, wav.data(), wav.size());
        return;
    }

    struct StreamState {
        SOCKET sock = INVALID_SOCKET;
        Response *resp = nullptr;
        std::string request_id;
        bool started = false;
    };
    auto state = std::make_shared<StreamState>();
    state->sock = sock;
    state->resp = &resp;
    state->request_id = job.id;

    auto sink = [state](const int16_t *data, int frames) {
        if (frames <= 0) return true;
        if (!state->started) {
            if (!send_stream_headers(state->resp, state->request_id)) return false;
            state->started = true;
        }
        return send_pcm_chunk(state->sock, data, static_cast<size_t>(frames) * 2 * sizeof(int16_t));
    };

    std::string submit_err;
    std::string job_id;
    if (!rt_.submit(job, sink, &job_id, &submit_err)) {
        resp.error(submit_err == "busy" ? 429 : 500, submit_err == "busy" ? "busy" : "submit_failed", submit_err);
        return;
    }
    JobInfo info;
    rt_.wait(job_id, kJobTimeoutMs, &info);
    if (!state->started) {
        // 一个字节都没发出去，还能改状态码
        if (info.state == JobState::Failed) {
            resp.error(500, "synth_failed", info.error);
        } else {
            resp.error(409, "cancelled", "任务在出声前被取消");
        }
        return;
    }
    send_all(sock, "0\r\n\r\n", 5);
}

void Server::handle_clone(SOCKET sock, const Request &req) {
    Response resp;
    resp.sock = sock;
    resp.cors_origin = allowed_origin(req);
    const auto query = parse_query(req.query);
    const auto it = query.find("name");
    if (it == query.end() || it->second.empty()) {
        resp.error(400, "bad_request", "缺少 name 参数");
        return;
    }
    if (req.body.size() < 44) {
        resp.error(400, "bad_request", "上传内容太小，不是 WAV");
        return;
    }
    const std::string ctype = lower(req.header("content-type"));
    if (ctype.find("audio/") == 0 || ctype.find("application/octet-stream") == 0) {
        // ok
    } else {
        resp.error(415, "unsupported_media_type", "需要 audio/wav 或 application/octet-stream");
        return;
    }
    std::string err;
    const std::vector<uint8_t> bytes(req.body.begin(), req.body.end());
    if (!rt_.clone_voice(it->second, bytes, &err)) {
        resp.error(400, "clone_failed", err);
        return;
    }
    const std::string payload = "{\"ok\":true,\"name\":\"" + json_escape(it->second) + "\"," +
                                voices_json(rt_) + "}";
    resp.raw_json(200, payload);
}

void Server::dispatch(SOCKET sock, Request &req) {
    Response resp;
    resp.sock = sock;

    if (req.method == "OPTIONS") {
        handle_options(sock, req);
        return;
    }
    if (!host_ok(req)) {
        resp.error(403, "bad_host", "Host 不被允许");
        return;
    }
    const std::string origin = lower(req.header("origin"));
    if (!origin.empty() && allowed_origin(req).empty()) {
        resp.error(403, "origin_denied", "该来源不被允许");
        return;
    }
    const std::string cors = allowed_origin(req);

    const std::string &path = req.path;
    if (path != "/api/v1/voices/clone" && req.body.size() > kMaxJsonBody) {
        resp.error(413, "payload_too_large", "JSON 请求体上限 256 KB");
        return;
    }
    if (path == "/api/v1/health") {
        resp.cors_origin = cors;
        resp.raw_json(200, "{\"ok\":true,\"version\":\"" MOSS_VERSION "\",\"api\":\"" MOSS_API_VERSION "\",\"queue\":" +
                               std::to_string(rt_.queue_depth()) + ",\"max\":" +
                               std::to_string(rt_.max_queue()) + "}");
        return;
    }
    if (path == "/api/v1/info") {
        resp.cors_origin = cors;
        resp.raw_json(200, rt_.info_json());
        return;
    }
    if (path == "/api/v1/voices") {
        resp.cors_origin = cors;
        resp.raw_json(200, "{" + voices_json(rt_) + "}");
        return;
    }
    if (path == "/api/v1/settings") {
        // 只有线程数：线程池随模型加载建好，改完必须重启进程才生效，所以这里只读写文件。
        resp.cors_origin = cors;
        EngineSettings st;
        load_settings(rt_.options().data_dir, &st);
        if (req.method == "GET") {
            // 返回"生效中"的线程数（自动模式下也是真实数字，别让页面看到 0+0 去猜）
            const int eff_llm = rt_.options().llm_threads;
            const int eff_decode = rt_.options().decode_threads;
            const bool auto_mode = st.llm_threads <= 0 && st.decode_threads <= 0;
            resp.raw_json(200, "{\"llm_threads\":" + std::to_string(eff_llm) +
                                   ",\"decode_threads\":" + std::to_string(eff_decode) +
                                   ",\"auto\":" + (auto_mode ? "true" : "false") +
                                   ",\"saved_llm_threads\":" + std::to_string(st.llm_threads) +
                                   ",\"saved_decode_threads\":" + std::to_string(st.decode_threads) +
                                   ",\"restart_required\":false}");
            return;
        }
        if (req.method != "POST") {
            resp.error(405, "method_not_allowed", "需要 GET 或 POST");
            return;
        }
        JsonValue root;
        std::string jerr;
        if (!json_parse(req.body, &root, &jerr) || !root.is_object()) {
            resp.error(400, "bad_json", "请求体不是 JSON 对象");
            return;
        }
        if (const JsonValue *v = root.get("llm_threads")) st.llm_threads = v->as_int(st.llm_threads);
        if (const JsonValue *v = root.get("decode_threads")) st.decode_threads = v->as_int(st.decode_threads);
        st.llm_threads = std::max(0, std::min(64, st.llm_threads));
        st.decode_threads = std::max(0, std::min(64, st.decode_threads));
        std::string werr;
        if (!save_settings(rt_.options().data_dir, st, &werr)) {
            resp.error(500, "save_failed", werr);
            return;
        }
        resp.raw_json(200, "{\"ok\":true,\"llm_threads\":" + std::to_string(st.llm_threads) +
                               ",\"decode_threads\":" + std::to_string(st.decode_threads) +
                               ",\"restart_required\":true}");
        return;
    }
    if (path == "/api/v1/restart") {
        // 网页上的「重启引擎」：起一个新进程接管，自己退出。端口由新进程的 --port-wait 等出来。
        if (req.method != "POST") {
            resp.error(405, "method_not_allowed", "需要 POST");
            return;
        }
        resp.cors_origin = cors;
        std::string rerr;
        if (!spawn_restart(&rerr)) {
            resp.error(500, "restart_failed", rerr);
            return;
        }
        resp.raw_json(200, "{\"ok\":true,\"restarting\":true}");
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        std::fflush(nullptr);
        ::ExitProcess(0);
    }
    if (path.rfind("/api/v1/jobs/", 0) == 0) {
        std::string rest = path.substr(std::string("/api/v1/jobs/").size());
        const bool cancel = rest.size() > 7 && rest.compare(rest.size() - 7, 7, "/cancel") == 0;
        if (cancel) rest.resize(rest.size() - 7);
        resp.cors_origin = cors;
        if (rest.empty()) {
            resp.error(400, "bad_request", "缺少 job id");
            return;
        }
        if (cancel) {
            if (req.method != "POST") {
                resp.error(405, "method_not_allowed", "cancel 需要 POST");
                return;
            }
            if (!rt_.cancel(rest)) {
                resp.error(404, "not_found", "没有这个任务");
                return;
            }
            resp.raw_json(200, "{\"ok\":true,\"id\":\"" + json_escape(rest) + "\"}");
            return;
        }
        JobInfo info;
        if (!rt_.job(rest, &info)) {
            resp.error(404, "not_found", "没有这个任务");
            return;
        }
        resp.raw_json(200, "{" + job_json(info) + "}");
        return;
    }
    if (path == "/api/v1/tts/stream") {
        if (req.method != "POST") {
            resp.error(405, "method_not_allowed", "需要 POST");
            return;
        }
        handle_tts(sock, req, false);
        return;
    }
    if (path == "/api/v1/tts/wav") {
        if (req.method != "POST") {
            resp.error(405, "method_not_allowed", "需要 POST");
            return;
        }
        handle_tts(sock, req, true);
        return;
    }
    if (path == "/api/v1/voices/clone") {
        if (req.method != "POST") {
            resp.error(405, "method_not_allowed", "需要 POST");
            return;
        }
        handle_clone(sock, req);
        return;
    }
    if (path.rfind("/api/", 0) == 0) {
        resp.cors_origin = cors;
        resp.error(404, "not_found", "没有这个接口");
        return;
    }
    if (req.method != "GET" && req.method != "HEAD") {
        resp.error(405, "method_not_allowed", "静态资源只支持 GET");
        return;
    }
    if (path == "/favicon.ico") {
        resp.headers(204, {{"Content-Length", "0"}});
        return;
    }
    send_file(resp, fs::path(opt_.web_dir), path, cors);
}

}  // namespace

bool serve_forever(Runtime &runtime, const ServerOptions &options, std::string *out_url, std::string *err) {
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        if (err) *err = "WSAStartup 失败";
        return false;
    }
    SetConsoleOutputCP(CP_UTF8);

    SOCKET listen_sock = INVALID_SOCKET;
    int port = 0;
    const int attempts = std::max(1, options.port_attempts);
    for (int i = 0; i < attempts; ++i) {
        const int candidate = options.port + i;
        // 首选端口可以等一下：网页上点"重启"时，新进程先起、旧进程后退出，
        // 新进程必须等旧进程松开 8759，否则会顺延到 8760 让页面地址漂移。
        const int tries = (i == 0 && options.port_wait_ms > 0)
                              ? std::max(1, options.port_wait_ms / 100)
                              : 1;
        for (int t = 0; t < tries && listen_sock == INVALID_SOCKET; ++t) {
            SOCKET s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (s == INVALID_SOCKET) break;
            // Windows 上没有 SO_REUSEADDR 的 Unix 语义：用它会让两个进程能绑同一端口。
            // 我们只要“本机唯一一个可信 TTS 服务”，所以用独占绑定。
            BOOL exclusive = TRUE;
            setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char *>(&exclusive),
                       sizeof(exclusive));
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(static_cast<unsigned short>(candidate));
            addr.sin_addr.s_addr = inet_addr(options.host.c_str());
            bool ok = ::bind(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0;
            if (ok) ok = ::listen(s, 16) == 0;
            if (ok) {
                listen_sock = s;
                port = candidate;
                break;
            }
            closesocket(s);
            if (t + 1 < tries) ::Sleep(100);
        }
        if (listen_sock != INVALID_SOCKET) break;
    }
    if (listen_sock == INVALID_SOCKET) {
        if (err) *err = "端口被占用：" + std::to_string(options.port) + "~" +
                        std::to_string(options.port + attempts - 1);
        return false;
    }

    const std::string url = "http://" + options.host + ":" + std::to_string(port) + "/";
    if (out_url) *out_url = url;
    std::printf("[serve] %s\n", url.c_str());
    std::printf("[serve] web   %s\n", options.web_dir.c_str());
    std::printf("[serve] 线程  llm=%d decode=%d，队列上限 %d\n", runtime.options().llm_threads,
                runtime.options().decode_threads, runtime.max_queue());
    std::fflush(stdout);
    if (options.open_browser) open_browser(url);

    Server server(runtime, options, port);
    std::atomic<int> active{0};
    while (true) {
        SOCKET client = ::accept(listen_sock, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;
        if (active.load() >= options.max_connections) {
            const char *busy = "HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
            send_all(client, busy, std::strlen(busy));
            closesocket(client);
            continue;
        }
        ++active;
        std::thread([&server, &active, client] {
            server.handle(client);
            --active;
        }).detach();
    }
}

}  // namespace moss
