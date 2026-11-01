// moss_decoder：整段音频解码 CLI
//   moss_decoder --weights decoder_f32.moss --codes codes.txt --out audio.f32 [--threads 4]
// codes.txt：每行 16 个整数（一帧），文件里有 T 行。
// 输出：交织立体声 float32 原始样本（L,R,L,R...）。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "moss/cpu_features.h"
#include "moss/decoder.h"
#include "moss/parallel.h"

using namespace moss;

int main(int argc, char **argv) {
    std::string weights, codes_path, out_path;
    int threads = 4;
    int stream_chunk = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--weights") weights = next();
        else if (a == "--codes") codes_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--stream") stream_chunk = std::stoi(next());
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            return 1;
        }
    }
    if (weights.empty() || codes_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "用法: moss_decoder --weights <x.moss> --codes <codes.txt> --out <audio.f32>\n");
        return 1;
    }

    const CpuFeatures &feat = cpu_features();
    std::fprintf(stderr, "cpu=%s avx2=%d backend=%s\n", feat.brand.c_str(), feat.avx2, feat.gemv_backend());

    std::vector<std::vector<int>> rows;
    {
        std::ifstream in(codes_path);
        if (!in) {
            std::fprintf(stderr, "打不开 codes 文件: %s\n", codes_path.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::vector<int> row;
            int v;
            while (iss >> v) row.push_back(v);
            if (row.size() != 16) {
                std::fprintf(stderr, "codes 行需要 16 个数，实际 %zu\n", row.size());
                return 1;
            }
            rows.push_back(std::move(row));
        }
    }
    const int frames = static_cast<int>(rows.size());
    if (frames <= 0) {
        std::fprintf(stderr, "codes 为空\n");
        return 1;
    }

    // 转成 [nq][T]
    std::vector<int32_t> codes(static_cast<size_t>(frames) * 16);
    for (int t = 0; t < frames; ++t) {
        for (int i = 0; i < 16; ++i) codes[static_cast<size_t>(i) * frames + t] = rows[t][i];
    }

    AudioDecoder decoder(threads);
    std::string err;
    const auto t_load0 = std::chrono::steady_clock::now();
    if (!decoder.load(weights, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

    std::vector<float> audio;
    std::vector<double> chunk_ms;
    if (stream_chunk > 0) {
        // 流式：按 S 帧一块，块间靠各段 KV 缓存衔接
        decoder.reset_stream();
        for (int start = 0; start < frames; start += stream_chunk) {
            const int n = std::min(stream_chunk, frames - start);
            std::vector<int32_t> c(static_cast<size_t>(n) * 16);
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < n; ++j) c[static_cast<size_t>(i) * n + j] = rows[start + j][i];
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> part = decoder.decode_chunk(c.data(), n);
            chunk_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            audio.insert(audio.end(), part.begin(), part.end());
        }
    } else {
        audio = decoder.decode(codes.data(), frames);
    }

    {
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(audio.data()), static_cast<std::streamsize>(audio.size() * sizeof(float)));
    }

    std::printf("{\n");
    std::printf("  \"frames\": %d,\n", frames);
    std::printf("  \"interleaved_samples\": %zu,\n", audio.size());
    std::printf("  \"per_channel_samples\": %zu,\n", audio.size() / 2);
    std::printf("  \"load_ms\": %.1f,\n", load_ms);
    std::printf("  \"decode_ms\": %.1f,\n", decoder.last_ms());
    std::printf("  \"stream_chunk\": %d,\n", stream_chunk);
    std::printf("  \"first_chunk_ms\": %.1f,\n", chunk_ms.empty() ? 0.0 : chunk_ms.front());
    std::printf("  \"chunks\": %zu,\n", chunk_ms.size());
    double stream_total = 0.0;
    for (double v : chunk_ms) stream_total += v;
    std::printf("  \"stream_total_ms\": %.1f,\n", stream_total);
    std::printf("  \"stream_rtf\": %.4f,\n", stream_total / (audio.size() / 2.0 / 48000.0 * 1000.0));
    std::printf("  \"stage_ms\": [");
    {
        const auto &sm = decoder.stage_ms();
        for (size_t i = 0; i < sm.size(); ++i) std::printf("%s%.1f", i ? ", " : "", sm[i]);
    }
    std::printf("],\n");
    float peak = 0.0f;
    for (float v : audio) peak = std::max(peak, std::fabs(v));
    std::printf("  \"peak\": %.4f\n", peak);
    std::printf("}\n");
    return 0;
}
