// moss_tts：产品 CLI（文本 -> 音频）。全链路在 TtsEngine 里：分词 -> prompt -> LLM 出码 -> 流式解码。
//
//   moss_tts --models models --text "你好，世界。" --out hello.wav
//   moss_tts --models models --text "..." --voice Xiaoyu --pcm-raw - > out.pcm
//   moss_tts --models models --profile voices/me.mossclone --text "..." --out out.wav
//
// --models 目录里应有：llm_q8x4.moss / decoder_f16.moss / encoder_f16.moss / tokenizer.mossvocab / voices/
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "moss/cpu_features.h"
#include "moss/engine.h"
#include "moss/http_server.h"
#include "moss/runtime.h"
#include "moss/stdio_server.h"
#include "moss/version.h"
#include "moss/settings.h"
#include "moss/voices.h"
#include "moss/wav.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace moss;

namespace {

std::string dir_join(const std::string &dir, const std::string &name) {
    if (dir.empty()) return name;
    if (dir.back() == '/' || dir.back() == '\\') return dir + name;
    return dir + "/" + name;
}

std::string exe_dir() {
#ifdef _WIN32
    wchar_t buf[4096];
    const DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(std::size(buf)));
    if (n == 0) return {};
    const int need = WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
    if (need <= 0) return {};
    std::string path(static_cast<size_t>(need), '\0');
    WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), path.data(), need, nullptr, nullptr);
    const size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
#else
    return ".";
#endif
}

std::string env_or_empty(const char *name) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

bool is_dir(const std::string &path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

// 模型目录查找顺序：--models 显式 > 程序同级的 models/ > %LOCALAPPDATA%\MossTTS\models
std::string resolve_models_dir(const std::string &explicit_dir) {
    if (!explicit_dir.empty()) return explicit_dir;
    const std::string beside = dir_join(exe_dir(), "models");
    if (is_dir(beside)) return beside;
#ifdef _WIN32
    const std::string local = env_or_empty("LOCALAPPDATA");
    if (!local.empty()) {
        const std::string shared = dir_join(local, "MossTTS/models");
        if (is_dir(shared)) return shared;
    }
#endif
    return "models";
}

std::string resolve_data_dir(const std::string &models_dir) {
    // 便携优先：模型在程序旁边时，缓存/用户音色也放旁边（删目录即干净卸载）
    const std::string beside = dir_join(exe_dir(), "models");
    if (models_dir == beside) return dir_join(exe_dir(), "userdata");
#ifdef _WIN32
    const std::string local = env_or_empty("LOCALAPPDATA");
    if (!local.empty()) return dir_join(local, "MossTTS");
#endif
    return dir_join(models_dir, "..");
}

bool file_exists(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

bool write_pcm16(const std::string &path, const std::vector<float> &interleaved) {
    std::vector<int16_t> pcm(interleaved.size());
    for (size_t i = 0; i < interleaved.size(); ++i) {
        const float v = std::max(-1.0f, std::min(1.0f, interleaved[i]));
        pcm[i] = static_cast<int16_t>(std::lrintf(v * 32767.0f));
    }
    if (path == "-") {
        std::fwrite(pcm.data(), sizeof(int16_t), pcm.size(), stdout);
        std::fflush(stdout);
        return true;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char *>(pcm.data()), static_cast<std::streamsize>(pcm.size() * 2));
    return static_cast<bool>(out);
}

void print_usage() {
    std::fprintf(stderr,
                 "用法:\n"
                 "  moss_tts [--serve] [--stdio] [--port 8759] [--port-wait ms] [--no-open] [--web dir] [--data dir]\n"
                 "           --models <dir> --text <文本> [--out x.wav] [--pcm-raw f|-]\n"
                 "      [--voice 音色 | --profile x.mossclone] [--threads 0=auto(2)] [--decode-threads 0=auto(2)]\n"
                 "      [--chunk 8] [--a16 off|decode|all] [--seed 0] [--greedy] [--frames 375] [--no-cache]\n"
                 "      [--temp 0.8] [--top-k 25] [--top-p 0.95] [--rep 1.2] [--greedy-frames 0]\n"
                 "      [--list-voices] [--selftest] [--version]\n"
                 "      --llm/--decoder/--encoder/--vocab/--voices/--cache 可覆盖 --models 下的默认路径\n"
                 "  不带参数 = 启动本地服务并打开浏览器；--stdio = 给插件用的 JSON 行通道。\n");
}

}  // namespace

int run(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string models;
    std::string llm, decoder, encoder, vocab, voices, cache;
    std::string text, text_file, out_path, raw_path, voice, profile;
    std::string a16 = "decode";  // 发布默认：吐 token 阶段 A16
    int threads = 0;             // 0 = 按物理核自动
    int decode_threads = 0;
    int chunk = 8;
    int frames = 375;
    int greedy_frames = 0;
    float temperature = 0.8f;
    int top_k = 25;
    float top_p = 0.95f;
    float rep = 1.2f;
    uint64_t seed = 0;
    bool greedy = false;
    bool use_cache = true;
    bool list_voices = false;
    bool selftest = false;
    bool serve = false;
    bool stdio_mode = false;
    bool open_browser = true;
    bool show_version = false;
    bool cache_explicit = false;
    int serve_port = 8759;
    int port_wait_ms = 0;
    std::string web_dir, data_dir, allow_origin;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--models") models = next();
        else if (a == "--llm") llm = next();
        else if (a == "--decoder") decoder = next();
        else if (a == "--encoder") encoder = next();
        else if (a == "--vocab") vocab = next();
        else if (a == "--voices") voices = next();
        else if (a == "--cache") { cache = next(); cache_explicit = true; }
        else if (a == "--text") text = next();
        else if (a == "--text-file") text_file = next();
        else if (a == "--out") out_path = next();
        else if (a == "--pcm-raw") raw_path = next();
        else if (a == "--voice") voice = next();
        else if (a == "--profile") profile = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--decode-threads") decode_threads = std::stoi(next());
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--a16") a16 = next();
        else if (a == "--frames") frames = std::stoi(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--greedy") greedy = true;
        else if (a == "--no-cache") use_cache = false;
        else if (a == "--greedy-frames") greedy_frames = std::stoi(next());
        else if (a == "--temp") temperature = std::stof(next());
        else if (a == "--top-k") top_k = std::stoi(next());
        else if (a == "--top-p") top_p = std::stof(next());
        else if (a == "--rep") rep = std::stof(next());
        else if (a == "--list-voices") list_voices = true;
        else if (a == "--selftest") selftest = true;
        else if (a == "--serve") serve = true;
        else if (a == "--stdio") stdio_mode = true;
        else if (a == "--port") serve_port = std::stoi(next());
        else if (a == "--no-open") open_browser = false;
        else if (a == "--port-wait") port_wait_ms = std::stoi(next());
        else if (a == "--web") web_dir = next();
        else if (a == "--data") data_dir = next();
        else if (a == "--allow-origin") allow_origin = next();
        else if (a == "--version") show_version = true;
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            print_usage();
            return 1;
        }
    }

    if (show_version) {
        std::printf("%s\n", MOSS_VERSION);
        return 0;
    }
    if (argc == 1) serve = true;  // 双击 = 起服务并打开浏览器
    models = resolve_models_dir(models);

    if (llm.empty()) llm = dir_join(models, "llm_q8x4.moss");
    if (decoder.empty()) decoder = dir_join(models, "decoder_f16.moss");
    if (encoder.empty()) encoder = dir_join(models, "encoder_f16.moss");
    if (vocab.empty()) vocab = dir_join(models, "tokenizer.mossvocab");
    if (voices.empty()) voices = dir_join(models, "voices");
    if (cache.empty() && use_cache) cache = dir_join(models, "cache");
    if (!text_file.empty()) {
        std::ifstream in(text_file, std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "打不开文本文件: %s\n", text_file.c_str());
            return 1;
        }
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    if (list_voices) {
        namespace fs = std::filesystem;
        std::error_code ec;
        std::vector<std::string> names;
        for (const auto &entry : fs::directory_iterator(voices, ec)) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (name.size() > 10 && name.compare(name.size() - 10, 10, ".mossclone") == 0) {
                names.push_back(name.substr(0, name.size() - 10));
            }
        }
        std::sort(names.begin(), names.end());
        const std::string def = pick_default_voice(voices);
        std::printf("{\"voices_dir\":\"%s\",\"count\":%zu,\"default\":\"%s\",\"voices\":[", voices.c_str(),
                    names.size(), def.c_str());
        for (size_t i = 0; i < names.size(); ++i) {
            const char *label = builtin_voice_label(names[i]);
            std::printf("%s{\"id\":\"%s\",\"label\":\"%s\"%s}", i ? "," : "", names[i].c_str(),
                        label ? label : names[i].c_str(), names[i] == def ? ",\"default\":true" : "");
        }
        std::printf("]}\n");
        return names.empty() ? 2 : 0;
    }
    if (!serve && !stdio_mode && text.empty() && !selftest) {
        print_usage();
        return 1;
    }
    if (!serve && !stdio_mode && !selftest && out_path.empty() && raw_path.empty()) {
        std::fprintf(stderr, "至少要给 --out 或 --pcm-raw 之一\n");
        return 1;
    }
    if (!serve && !stdio_mode && threads <= 0) threads = 2;
    if (!serve && !stdio_mode && decode_threads <= 0) decode_threads = threads;
    // W8A16：off / decode / all（见 gemv.cpp 的 act16_enabled）
    _putenv_s("MOSS_A16", a16 == "off" ? "" : a16.c_str());
    if (!file_exists(llm)) {
        std::fprintf(stderr, "找不到 LLM 权重: %s\n", llm.c_str());
        return 1;
    }
    if (!file_exists(decoder)) {
        std::fprintf(stderr, "找不到解码器权重: %s\n", decoder.c_str());
        return 1;
    }
    if (!file_exists(vocab)) {
        std::fprintf(stderr, "找不到词表: %s（用 tools/spm_dump.py 生成）\n", vocab.c_str());
        return 1;
    }

    const CpuFeatures &feat = cpu_features();
    std::fprintf(stderr, "cpu=%s avx2=%d backend=%s llm_threads=%d decode_threads=%d\n", feat.brand.c_str(),
                 feat.avx2, feat.gemv_backend(), threads, decode_threads);

    if (serve || stdio_mode) {
        std::string serve_err;
        const std::string resolved_data = data_dir.empty() ? resolve_data_dir(models) : data_dir;
        const std::string user_voices = dir_join(resolved_data, "voices/user");
        // 常驻模式（--serve / --stdio）：缓存统一放用户数据目录，程序目录保持只读也能跑
        if (!cache_explicit) cache = dir_join(resolved_data, "cache");
        std::error_code ec;
        std::filesystem::create_directories(user_voices, ec);
        std::filesystem::create_directories(cache, ec);

        RuntimeOptions ro;
        ro.llm_path = llm;
        ro.decoder_path = decoder;
        ro.encoder_path = encoder;
        ro.tokenizer_path = vocab;
        ro.voices_dir = voices;
        ro.user_voices_dir = user_voices;
        ro.cache_dir = cache;
        ro.data_dir = resolved_data;
        ro.manifest_path = dir_join(models, "manifest.json");
        // 线程数优先级：命令行 > <data>/settings.json > 自动（2+2）
        EngineSettings settings;
        load_settings(resolved_data, &settings);
        if (threads <= 0 && settings.llm_threads > 0) threads = settings.llm_threads;
        if (decode_threads <= 0 && settings.decode_threads > 0) decode_threads = settings.decode_threads;
        ro.llm_threads = threads;  // 0 = 自动
        ro.decode_threads = decode_threads;

        Runtime runtime;
        if (!runtime.load(ro, &serve_err)) {
            std::fprintf(stderr, "%s\n", serve_err.c_str());
            return 1;
        }
        std::fprintf(stderr, "data=%s\n", resolved_data.c_str());

        if (stdio_mode) return run_stdio_server(runtime);

        ServerOptions so;
        so.web_dir = web_dir.empty() ? dir_join(exe_dir(), "web") : web_dir;
        so.port = serve_port;
        so.port_wait_ms = port_wait_ms;
        so.open_browser = open_browser;
        if (!allow_origin.empty()) {
            // `--allow-origin none` = 完全禁止跨源（连项目门户页也不放行）
            if (allow_origin == "none") {
                so.allow_project_origin = false;
            } else {
                so.allow_origins.push_back(allow_origin);
            }
        }
        std::string url;
        if (!serve_forever(runtime, so, &url, &serve_err)) {
            std::fprintf(stderr, "%s\n", serve_err.c_str());
            return 1;
        }
        return 0;
    }

    EngineOptions options;
    options.llm_path = llm;
    options.decoder_path = decoder;
    options.encoder_path = encoder;
    options.tokenizer_path = vocab;
    options.voices_dir = voices;
    options.cache_dir = cache;
    options.llm_threads = threads;
    options.decode_threads = decode_threads;

    TtsEngine engine;
    std::string err;
    if (!engine.load(options, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (!profile.empty()) {
        if (!engine.set_voice(profile, &err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
    } else if (!engine.set_voice(voice.empty() ? pick_default_voice(voices) : voice, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    if (selftest) {
        const std::string probe = "你好，这是一次自检。";
        std::vector<float> audio;
        SynthOptions probe_options;
        probe_options.seed = 0;
        probe_options.greedy = true;
        probe_options.max_frames = 16;
        probe_options.chunk_frames = chunk;
        probe_options.on_audio = [&](const float *data, int frames) {
            audio.insert(audio.end(), data, data + static_cast<size_t>(frames) * 2);
        };
        SynthStats stats;
        const bool ok = engine.synthesize(probe, probe_options, &stats, &err);
        float peak = 0.0f;
        for (float v : audio) peak = std::max(peak, std::fabs(v));
        const bool pass = ok && stats.frames > 0 && peak > 0.01f;
        std::printf("{\"selftest\":\"%s\",\"frames\":%d,\"peak\":%.3f,\"text_tokens\":%d}\n",
                    pass ? "PASS" : "FAIL", stats.frames, peak, stats.text_tokens);
        return pass ? 0 : 2;
    }

    std::vector<float> audio;
    SynthOptions synth;
    synth.seed = seed;
    synth.greedy = greedy;
    synth.max_frames = frames > 0 ? frames : 375;
    synth.chunk_frames = chunk;
    synth.greedy_frames = greedy_frames;
    synth.temperature = temperature;
    synth.top_k = top_k;
    synth.top_p = top_p;
    synth.repetition_penalty = rep;
    synth.greedy_frames = greedy_frames;
    synth.on_audio = [&](const float *data, int n) {
        audio.insert(audio.end(), data, data + static_cast<size_t>(n) * 2);
    };
    SynthStats stats;
    if (!engine.synthesize(text, synth, &stats, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    if (!out_path.empty()) {
        if (!wav_write16(out_path, audio, 48000, 2, &err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
    }
    if (!raw_path.empty()) {
        if (!write_pcm16(raw_path, audio)) {
            std::fprintf(stderr, "写不了 PCM: %s\n", raw_path.c_str());
            return 1;
        }
    }

    std::printf("{\n");
    std::printf("  \"voice\": \"%s\",\n", engine.voice().name.c_str());
    std::printf("  \"voice_frames\": %d,\n", engine.voice().frames);
    std::printf("  \"text_chars\": %zu,\n", text.size());
    std::printf("  \"text_tokens\": %d,\n", stats.text_tokens);
    std::printf("  \"chunks\": %d,\n", stats.chunks);
    std::printf("  \"frames\": %d,\n", stats.frames);
    std::printf("  \"prefix_ms\": %.1f,\n", stats.prefix_ms);
    std::printf("  \"prefill_ms\": %.1f,\n", stats.prefill_ms);
    std::printf("  \"decode_ms\": %.1f,\n", stats.decode_ms);
    std::printf("  \"first_audio_ms\": %.1f,\n", stats.first_audio_ms);
    std::printf("  \"audio_seconds\": %.3f,\n", stats.audio_seconds);
    std::printf("  \"total_ms\": %.1f,\n", stats.total_ms);
    std::printf("  \"prefix_cache_hit\": %s,\n", stats.prefix_cache_hit ? "true" : "false");
    std::printf("  \"rtf\": %.4f,\n", stats.total_ms / std::max(1e-3, stats.audio_seconds * 1000.0));
    std::printf("  \"greedy\": %s,\n", greedy ? "true" : "false");
    std::printf("  \"seed\": %llu\n", static_cast<unsigned long long>(seed));
    std::printf("}\n");
    return 0;
}

#ifdef _WIN32
// 命令行参数按 UTF-16 传进来，C 运行时的 main 会转成 ANSI（GBK），中文会被打碎。
// 这里用 wmain 取宽字符再转 UTF-8，保证 --text 里的中文不走样。
#include <windows.h>

int wmain(int argc, wchar_t **argv) {
    std::vector<std::string> storage;
    storage.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        const int need = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(static_cast<size_t>(need > 0 ? need - 1 : 0), '\0');
        if (need > 1) WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, s.data(), need, nullptr, nullptr);
        storage.push_back(std::move(s));
    }
    std::vector<char *> ptrs;
    ptrs.reserve(storage.size());
    for (std::string &s : storage) ptrs.push_back(s.data());
    return run(argc, ptrs.data());
}
#else
int main(int argc, char **argv) { return run(argc, argv); }
#endif