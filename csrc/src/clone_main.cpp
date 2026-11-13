// moss_clone：把一段参考音频编码成音色档案（.mossclone）。
//
//   moss_clone --encoder weights/encoder_f16.moss --in ref.wav --out voices/me.mossclone ^
//              [--start 秒] [--seconds 秒] [--threads 4] [--name 我的声音] [--dump-codes codes.txt]
//
// 输入支持 PCM16/24/32、float32、WAVE_FORMAT_EXTENSIBLE；任意采样率（内部重采样到 48 kHz），
// 单声道会复制成双声道。压缩格式（MP3-in-WAV 之类）不收：浏览器那边先解码成 PCM。
// 参考音频最长取 20 秒（--start 选起点，--seconds 选长度），超出的部分自动截掉。
// 输出是 16 路码（frame-major），供 moss_tts 当 prompt 前缀用。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "moss/cpu_features.h"
#include "moss/encoder.h"
#include "moss/profile.h"
#include "moss/resample.h"
#include "moss/wav.h"

using namespace moss;

namespace {

using Clock = std::chrono::steady_clock;

// 参考音频上限：再长只会拉长 prompt、变慢，音色不会更好（和 engine.clone_audio 一致）。
constexpr double kMaxCloneSeconds = 20.0;
constexpr double kCloneTolerance = 0.05;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char **argv) {
    std::string encoder_path, in_path, out_path, name, codes_out;
    int threads = 4;
    double start_sec = 0.0;
    double seconds_sec = 0.0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--encoder") encoder_path = next();
        else if (a == "--in") in_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--name") name = next();
        else if (a == "--start") start_sec = std::atof(next().c_str());
        else if (a == "--seconds") seconds_sec = std::atof(next().c_str());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--dump-codes") codes_out = next();
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            return 1;
        }
    }
    if (encoder_path.empty() || in_path.empty() || out_path.empty()) {
        std::fprintf(stderr,
                     "用法: moss_clone --encoder <encoder.moss> --in <ref.wav> --out <x.mossclone> "
                     "[--start 秒] [--seconds 秒] [--threads T] [--name 名字] [--dump-codes codes.txt]\n");
        return 1;
    }

    const CpuFeatures &feat = cpu_features();
    std::fprintf(stderr, "cpu=%s avx2=%d backend=%s threads=%d\n", feat.brand.c_str(), feat.avx2,
                 feat.gemv_backend(), threads);

    // 1) 读音频
    std::vector<float> pcm;
    int sample_rate = 0;
    int channels = 0;
    std::string err;
    if (!wav_read(in_path, &pcm, &sample_rate, &channels, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    int in_frames = static_cast<int>(pcm.size() / static_cast<size_t>(channels));

    // 2) 声道：单声道复制成双声道；>2 取前两路
    if (channels == 1) {
        std::vector<float> stereo(static_cast<size_t>(in_frames) * 2);
        for (int i = 0; i < in_frames; ++i) {
            stereo[static_cast<size_t>(i) * 2] = pcm[static_cast<size_t>(i)];
            stereo[static_cast<size_t>(i) * 2 + 1] = pcm[static_cast<size_t>(i)];
        }
        pcm.swap(stereo);
        channels = 2;
    } else if (channels > 2) {
        std::vector<float> stereo(static_cast<size_t>(in_frames) * 2);
        for (int i = 0; i < in_frames; ++i) {
            stereo[static_cast<size_t>(i) * 2] = pcm[static_cast<size_t>(i) * channels];
            stereo[static_cast<size_t>(i) * 2 + 1] = pcm[static_cast<size_t>(i) * channels + 1];
        }
        pcm.swap(stereo);
        std::fprintf(stderr, "输入是 %d 声道，只取前两路\n", channels);
        channels = 2;
    }

    // 3) 选段：--start 起，--seconds 长；超过 20 s 的部分截掉（只有前 20 s 有用）
    const double source_seconds = static_cast<double>(in_frames) / static_cast<double>(sample_rate);
    int begin = static_cast<int>(std::max(0.0, start_sec) * sample_rate);
    if (begin > in_frames) begin = in_frames;
    int take = in_frames - begin;
    if (seconds_sec > 0) take = std::min(take, static_cast<int>(seconds_sec * sample_rate + 0.5));
    const int cap = static_cast<int>(kMaxCloneSeconds * sample_rate + 0.5);
    if (take > cap) take = cap;
    if (take <= 0) {
        std::fprintf(stderr, "选出来的参考音频是空的（输入 %.2f s，起点 %.2f s）\n", source_seconds, start_sec);
        return 1;
    }
    if (begin != 0 || take != in_frames) {
        std::vector<float> cut(static_cast<size_t>(take) * static_cast<size_t>(channels));
        std::memcpy(cut.data(), pcm.data() + static_cast<size_t>(begin) * static_cast<size_t>(channels),
                    cut.size() * sizeof(float));
        pcm.swap(cut);
        in_frames = take;
        if (source_seconds > kMaxCloneSeconds + kCloneTolerance && seconds_sec <= 0) {
            std::fprintf(stderr, "输入 %.1f s，只取前 %.0f s（--start 可以换一段）\n", source_seconds,
                         kMaxCloneSeconds);
        }
    }

    // 4) 重采样到 48 kHz（编码器要求）
    double resample_ms = 0.0;
    if (sample_rate != 48000) {
        const auto t0 = Clock::now();
        pcm = resample_interleaved(pcm.data(), in_frames, channels, sample_rate, 48000);
        resample_ms = ms_since(t0);
        sample_rate = 48000;
    }

    // 5) 编码
    AudioEncoder encoder(threads);
    const auto t_load0 = Clock::now();
    if (!encoder.load(encoder_path, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const double load_ms = ms_since(t_load0);

    std::vector<int32_t> codes;
    const auto t_enc0 = Clock::now();
    const int frames = encoder.encode(pcm.data(), static_cast<int>(pcm.size() / 2), &codes);
    const double encode_ms = ms_since(t_enc0);
    if (frames <= 0) {
        std::fprintf(stderr, "参考音频太短（至少要 80 ms）\n");
        return 1;
    }

    // 6) 落盘
    VoiceProfile profile;
    profile.name = name;
    profile.frames = frames;
    profile.sample_rate = sample_rate;
    profile.codes.resize(static_cast<size_t>(frames) * 16);
    for (size_t i = 0; i < profile.codes.size(); ++i) profile.codes[i] = static_cast<int16_t>(codes[i]);
    if (!profile_save(out_path, profile, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (!codes_out.empty()) {
        std::ofstream cf(codes_out);
        for (int f = 0; f < frames; ++f) {
            for (int ch = 0; ch < 16; ++ch) cf << codes[static_cast<size_t>(f) * 16 + ch] << (ch + 1 < 16 ? " " : "");
            cf << "\n";
        }
    }

    std::printf("{\n");
    std::printf("  \"in\": \"%s\",\n", in_path.c_str());
    std::printf("  \"out\": \"%s\",\n", out_path.c_str());
    std::printf("  \"name\": \"%s\",\n", name.c_str());
    std::printf("  \"input_rate\": %d,\n", sample_rate);
    std::printf("  \"frames\": %d,\n", frames);
    std::printf("  \"seconds\": %.2f,\n", static_cast<double>(frames) * 0.08);
    std::printf("  \"load_ms\": %.1f,\n", load_ms);
    std::printf("  \"resample_ms\": %.1f,\n", resample_ms);
    std::printf("  \"encode_ms\": %.1f,\n", encode_ms);
    std::printf("  \"stage_ms\": [");
    const auto &st = encoder.stage_ms();
    for (size_t i = 0; i < st.size(); ++i) std::printf("%s%.0f", i ? ", " : "", st[i]);
    std::printf("],\n");
    std::printf("  \"profile_bytes\": %zu\n", static_cast<size_t>(frames) * 16 * 2 + 28 + name.size());
    std::printf("}\n");
    return 0;
}
