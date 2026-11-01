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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "moss/cpu_features.h"
#include "moss/engine.h"
#include "moss/wav.h"

using namespace moss;

namespace {

std::string dir_join(const std::string &dir, const std::string &name) {
    if (dir.empty()) return name;
    if (dir.back() == '/' || dir.back() == '\\') return dir + name;
    return dir + "/" + name;
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
                 "用法: moss_tts --models <dir> --text <文本> [--out x.wav] [--pcm-raw f|-]\n"
                 "      [--voice Junhao | --profile x.mossclone] [--threads 4] [--decode-threads 6]\n"
                 "      [--chunk 8] [--seed 0] [--greedy] [--frames 375] [--no-cache]\n"
                 "      [--list-voices] [--selftest]\n"
                 "      --llm/--decoder/--encoder/--vocab/--voices/--cache 可覆盖 --models 下的默认路径\n");
}

}  // namespace

int run(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string models = "models";
    std::string llm, decoder, encoder, vocab, voices, cache;
    std::string text, text_file, out_path, raw_path, voice = "Junhao", profile;
    int threads = 4;
    int decode_threads = 0;
    int chunk = 8;
    int frames = 375;
    int greedy_frames = 1;
    uint64_t seed = 0;
    bool greedy = false;
    bool use_cache = true;
    bool list_voices = false;
    bool selftest = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--models") models = next();
        else if (a == "--llm") llm = next();
        else if (a == "--decoder") decoder = next();
        else if (a == "--encoder") encoder = next();
        else if (a == "--vocab") vocab = next();
        else if (a == "--voices") voices = next();
        else if (a == "--cache") cache = next();
        else if (a == "--text") text = next();
        else if (a == "--text-file") text_file = next();
        else if (a == "--out") out_path = next();
        else if (a == "--pcm-raw") raw_path = next();
        else if (a == "--voice") voice = next();
        else if (a == "--profile") profile = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--decode-threads") decode_threads = std::stoi(next());
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--frames") frames = std::stoi(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--greedy") greedy = true;
        else if (a == "--no-cache") use_cache = false;
        else if (a == "--greedy-frames") greedy_frames = std::stoi(next());
        else if (a == "--list-voices") list_voices = true;
        else if (a == "--selftest") selftest = true;
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            print_usage();
            return 1;
        }
    }

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
        std::printf("{\"voices_dir\":\"%s\",\"count\":%zu,\"voices\":[", voices.c_str(), names.size());
        for (size_t i = 0; i < names.size(); ++i) std::printf("%s\"%s\"", i ? "," : "", names[i].c_str());
        std::printf("]}\n");
        return names.empty() ? 2 : 0;
    }
    if (text.empty() && !selftest) {
        print_usage();
        return 1;
    }
    if (!selftest && out_path.empty() && raw_path.empty()) {
        std::fprintf(stderr, "至少要给 --out 或 --pcm-raw 之一\n");
        return 1;
    }
    if (decode_threads <= 0) decode_threads = threads;
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
    } else if (!engine.set_voice(voice, &err)) {
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