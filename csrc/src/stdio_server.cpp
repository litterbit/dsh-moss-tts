#include "moss/stdio_server.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "moss/json.h"
#include "moss/runtime.h"
#include "moss/version.h"

namespace moss {
namespace {

std::mutex g_out_mutex;

std::string base64_encode(const unsigned char *data, size_t len) {
    static const char *kTable = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        const unsigned v0 = data[i];
        const unsigned v1 = i + 1 < len ? data[i + 1] : 0;
        const unsigned v2 = i + 2 < len ? data[i + 2] : 0;
        const unsigned triple = (v0 << 16) | (v1 << 8) | v2;
        out.push_back(kTable[(triple >> 18) & 0x3F]);
        out.push_back(kTable[(triple >> 12) & 0x3F]);
        out.push_back(i + 1 < len ? kTable[(triple >> 6) & 0x3F] : '=');
        out.push_back(i + 2 < len ? kTable[triple & 0x3F] : '=');
    }
    return out;
}

bool base64_decode(const std::string &text, std::vector<uint8_t> *out) {
    static int table[256];
    static bool ready = false;
    if (!ready) {
        for (int i = 0; i < 256; ++i) table[i] = -1;
        const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) table[static_cast<unsigned char>(alphabet[i])] = i;
        ready = true;
    }
    out->clear();
    out->reserve(text.size() / 4 * 3);
    unsigned buffer = 0;
    int bits = 0;
    for (const char ch : text) {
        if (ch == '=' || ch == '\n' || ch == '\r' || ch == ' ') continue;
        const int value = table[static_cast<unsigned char>(ch)];
        if (value < 0) return false;
        buffer = (buffer << 6) | static_cast<unsigned>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out->push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
        }
    }
    return true;
}

void write_line(const std::string &line) {
    std::lock_guard<std::mutex> lock(g_out_mutex);
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::string make_error(const std::string &id, const std::string &code, const std::string &message) {
    return "{\"id\":\"" + json_escape(id) + "\",\"type\":\"error\",\"code\":\"" + json_escape(code) +
           "\",\"error\":\"" + json_escape(message) + "\"}";
}

std::string job_line(const std::string &id, const JobInfo &info) {
    std::string out = "{\"id\":\"" + json_escape(id) + "\",\"type\":\"done\",\"job\":{";
    out += "\"id\":\"" + json_escape(info.id) + "\"";
    out += ",\"state\":\"";
    out += job_state_name(info.state);
    out += "\",\"frames\":" + std::to_string(info.frames);
    out += ",\"audio_ms\":" + std::to_string(static_cast<long long>(info.audio_ms));
    out += ",\"compute_ms\":" + std::to_string(static_cast<long long>(info.compute_ms));
    out += ",\"first_pcm_ms\":" + std::to_string(static_cast<long long>(info.first_pcm_ms));
    out += ",\"rtf\":" + std::to_string(info.rtf);
    out += ",\"text_tokens\":" + std::to_string(info.text_tokens);
    out += ",\"cache_hit\":";
    out += info.cache_hit ? "true" : "false";
    out += ",\"error\":\"" + json_escape(info.error) + "\"}}";
    return out;
}

bool parse_request(const JsonValue &root, JobRequest *request, std::string *err) {
    if (!root.is_object()) {
        *err = "请求必须是 JSON 对象";
        return false;
    }
    const JsonValue *text = root.get("text");
    if (text == nullptr || text->as_string().empty()) {
        *err = "text 不能为空";
        return false;
    }
    request->text = text->as_string();
    if (const JsonValue *v = root.get("voice")) request->voice = v->as_string();
    if (const JsonValue *v = root.get("seed")) request->seed = static_cast<uint64_t>(v->as_number(0));
    if (const JsonValue *v = root.get("greedy")) request->greedy = v->as_bool(false);
    if (const JsonValue *v = root.get("max_frames")) request->max_frames = v->as_int(375);
    if (const JsonValue *v = root.get("chunk_frames")) request->chunk_frames = v->as_int(8);
    if (const JsonValue *v = root.get("min_frames")) request->min_frames = v->as_int(0);
    if (const JsonValue *v = root.get("temperature")) request->temperature = static_cast<float>(v->as_number(0));
    if (const JsonValue *v = root.get("top_k")) request->top_k = v->as_int(0);
    if (const JsonValue *v = root.get("top_p")) request->top_p = static_cast<float>(v->as_number(0));
    if (const JsonValue *v = root.get("repetition_penalty"))
        request->repetition_penalty = static_cast<float>(v->as_number(0));
    return true;
}

}  // namespace

int run_stdio_server(Runtime &runtime) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        JsonValue root;
        std::string parse_err;
        if (!json_parse(line, &root, &parse_err)) {
            write_line(make_error("", "bad_json", parse_err));
            continue;
        }
        const std::string id = root.get("id") ? root.get("id")->as_string() : std::string();
        const std::string cmd = root.get("cmd") ? root.get("cmd")->as_string() : std::string();

        if (cmd == "ping") {
            write_line("{\"id\":\"" + json_escape(id) + "\",\"type\":\"pong\",\"version\":\"" MOSS_VERSION "\"}");
        } else if (cmd == "info") {
            write_line("{\"id\":\"" + json_escape(id) + "\",\"type\":\"info\",\"info\":" + runtime.info_json() + "}");
        } else if (cmd == "voices") {
            const std::string def = runtime.default_voice();
            auto one = [&](const VoiceInfo &v) {
                return "{\"id\":\"" + json_escape(v.id) + "\",\"label\":\"" + json_escape(v.label) +
                       "\",\"frames\":" + std::to_string(v.frames) + ",\"seconds\":" +
                       std::to_string(v.seconds) + ",\"default\":" + (v.id == def ? "true" : "false") + "}";
            };
            std::string out = "{\"id\":\"" + json_escape(id) + "\",\"type\":\"voices\",\"builtin\":[";
            const auto voices = runtime.voices();
            bool first = true;
            for (const auto &v : voices) {
                if (v.user) continue;
                if (!first) out += ",";
                first = false;
                out += one(v);
            }
            out += "],\"user\":[";
            first = true;
            for (const auto &v : voices) {
                if (!v.user) continue;
                if (!first) out += ",";
                first = false;
                out += one(v);
            }
            out += "],\"default\":\"" + json_escape(def) + "\"}";
            write_line(out);
        } else if (cmd == "cancel") {
            const std::string job = root.get("job") ? root.get("job")->as_string() : std::string();
            const bool ok = !job.empty() && runtime.cancel(job);
            write_line("{\"id\":\"" + json_escape(id) + "\",\"type\":\"cancelled\",\"ok\":" +
                       (ok ? "true" : "false") + ",\"job\":\"" + json_escape(job) + "\"}");
        } else if (cmd == "clone") {
            const std::string name = root.get("name") ? root.get("name")->as_string() : std::string();
            const std::string b64 = root.get("wav") ? root.get("wav")->as_string() : std::string();
            std::vector<uint8_t> bytes;
            std::string err;
            if (name.empty() || b64.empty() || !base64_decode(b64, &bytes)) {
                write_line(make_error(id, "bad_request", "需要 name 与 base64 的 wav"));
            } else if (!runtime.clone_voice(name, bytes, &err)) {
                write_line(make_error(id, "clone_failed", err));
            } else {
                write_line("{\"id\":\"" + json_escape(id) + "\",\"type\":\"cloned\",\"name\":\"" +
                           json_escape(name) + "\"}");
            }
        } else if (cmd == "speak") {
            JobRequest request;
            std::string err;
            if (!parse_request(root, &request, &err)) {
                write_line(make_error(id, "bad_request", err));
                continue;
            }
            // 用客户端给的 id 当 job id：这样可以先发 begin（格式字段），再 submit，
            // 避免工作线程抢在 begin 前面吐 audio 行。
            static std::atomic<uint64_t> counter{0};
            request.id = id.empty() ? "s" + std::to_string(++counter) : id;
            // 音色解析与 HTTP 一致：只接受 id + scope，路径由服务端拼；不给 voice 就用默认音色
            const std::string scope = root.get("scope") ? root.get("scope")->as_string("builtin") : "builtin";
            if (request.voice.empty()) request.voice = runtime.default_voice();
            const std::string base =
                scope == "user" ? runtime.options().user_voices_dir : runtime.options().voices_dir;
            request.profile = base + "/" + request.voice + ".mossclone";

            write_line("{\"id\":\"" + json_escape(id) + "\",\"type\":\"begin\",\"job\":\"" +
                       json_escape(request.id) + "\",\"format\":{\"encoding\":\"s16le\",\"rate\":48000,\"channels\":2}}");

            auto id_holder = std::make_shared<std::string>(id);
            auto sink = [id_holder](const int16_t *pcm, int frames) {
                const std::string encoded =
                    base64_encode(reinterpret_cast<const unsigned char *>(pcm),
                                  static_cast<size_t>(frames) * 2 * sizeof(int16_t));
                write_line("{\"id\":\"" + json_escape(*id_holder) + "\",\"type\":\"audio\",\"pcm\":\"" + encoded +
                           "\"}");
                return true;
            };
            std::string job_id;
            if (!runtime.submit(request, sink, &job_id, &err)) {
                write_line(make_error(id, err == "busy" ? "busy" : "submit_failed", err));
                continue;
            }
            // 单独线程等结果：主线程要继续读 stdin，否则 cancel 永远进不来
            std::thread([&runtime, id, job_id] {
                JobInfo info;
                runtime.wait(job_id, 600000, &info);
                write_line(job_line(id, info));
            }).detach();
        } else {
            write_line(make_error(id, "unknown_cmd", "未知的 cmd: " + cmd));
        }
    }
    return 0;
}

}  // namespace moss
