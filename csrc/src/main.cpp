// moss_llm：C++ LLM 推理 CLI
//   moss_llm --weights x.moss --prompt prompt.txt --frames 8 [--greedy] [--seed 0]
//            [--out codes.txt] [--dump-hidden hidden.bin]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "moss/cpu_features.h"
#include "moss/llm.h"
#include "moss/parallel.h"

using namespace moss;

namespace {

bool read_prompt(const std::string &path, std::vector<Row> *rows, std::string *err) {
    std::ifstream in(path);
    if (!in) {
        *err = "打不开 prompt 文件: " + path;
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Row row{};
        for (int i = 0; i < kNVq + 1; ++i) {
            if (!(iss >> row[i])) {
                *err = "prompt 行格式不对（需要 17 个整数）: " + line;
                return false;
            }
        }
        rows->push_back(row);
    }
    return !rows->empty();
}

}  // namespace

int main(int argc, char **argv) {
    std::string weights, prompt_path, out_path, hidden_path, force_path, logits_path;
    std::string gate_path, end_mode = "greedy";
    int frames = 8;
    int threads = 4;
    int min_frames = 0;
    bool greedy = false;
    uint64_t seed = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--weights") weights = next();
        else if (a == "--prompt") prompt_path = next();
        else if (a == "--frames") frames = std::stoi(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--out") out_path = next();
        else if (a == "--dump-hidden") hidden_path = next();
        else if (a == "--force-codes") force_path = next();
        else if (a == "--dump-logits") logits_path = next();
        else if (a == "--dump-gate") gate_path = next();
        else if (a == "--end-mode") end_mode = next();
        else if (a == "--min-frames") min_frames = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--greedy") greedy = true;
        else if (a == "--sample") greedy = false;
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            return 1;
        }
    }
    if (weights.empty() || prompt_path.empty()) {
        std::fprintf(stderr,
                     "用法: moss_llm --weights <x.moss> --prompt <prompt.txt> [--frames N] "
                     "[--greedy|--sample] [--seed S] [--out codes.txt] [--dump-hidden h.bin]\n");
        return 1;
    }

    std::vector<Row> rows;
    std::string err;
    if (!read_prompt(prompt_path, &rows, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    const CpuFeatures &feat = cpu_features();
    std::fprintf(stderr, "cpu=%s avx2=%d backend=%s threads=%d\n", feat.brand.c_str(), feat.avx2,
                 feat.gemv_backend(), threads);

    LlmModel model(threads);
    const auto t_load0 = std::chrono::steady_clock::now();
    if (!model.load(weights, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

    const auto t_pre0 = std::chrono::steady_clock::now();
    model.prefill(rows);
    const double prefill_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pre0).count();

    if (!hidden_path.empty()) {
        std::ofstream hf(hidden_path, std::ios::binary);
        hf.write(reinterpret_cast<const char *>(model.hidden()), sizeof(float) * kHidden);
    }

    SampleParams params;
    params.greedy = greedy;
    params.end_greedy = end_mode != "sample";
    params.min_frames = min_frames;
    // 上游 manifest 的默认采样参数（audio 头）；--greedy 时全部忽略
    params.temperature = 0.8f;
    params.top_k = 25;
    params.top_p = 0.95f;
    params.repetition_penalty = 1.2f;

    if (!logits_path.empty()) model.set_logits_dump(logits_path);
    if (!gate_path.empty()) model.set_gate_dump(gate_path);

    std::vector<FrameTiming> timing;
    std::vector<Row> forced;
    bool has_forced = false;
    if (!force_path.empty()) {
        std::ifstream in(force_path);
        if (!in) {
            std::fprintf(stderr, "打不开强制 codes 文件: %s\n", force_path.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            Row row{};
            row[0] = kSlotToken;
            for (int ch = 0; ch < kNVq; ++ch) iss >> row[ch + 1];
            forced.push_back(row);
        }
        has_forced = !forced.empty();
        if (has_forced && static_cast<int>(forced.size()) < frames) frames = static_cast<int>(forced.size());
    }

    const auto t_gen0 = std::chrono::steady_clock::now();
    const std::vector<Row> codes =
        model.generate(frames, params, seed, &timing, has_forced ? &forced : nullptr);
    const double gen_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_gen0).count();

    long match = 0;
    long total = 0;
    if (has_forced) {
        for (size_t i = 0; i < codes.size() && i < forced.size(); ++i) {
            for (int ch = 0; ch < kNVq; ++ch) {
                ++total;
                if (codes[i][ch + 1] == forced[i][ch + 1]) ++match;
            }
        }
    }

    std::printf("{\n");
    std::printf("  \"weights\": \"%s\",\n", weights.c_str());
    std::printf("  \"prompt_rows\": %d,\n", static_cast<int>(rows.size()));
    std::printf("  \"load_ms\": %.1f,\n", load_ms);
    std::printf("  \"prefill_ms\": %.1f,\n", prefill_ms);
    std::printf("  \"frames\": %d,\n", static_cast<int>(codes.size()));
    std::printf("  \"gen_ms\": %.1f,\n", gen_ms);
    if (!timing.empty()) {
        double g = 0.0, l = 0.0;
        for (const auto &t : timing) {
            g += t.global_ms;
            l += t.local_ms;
        }
        std::printf("  \"global_ms_per_frame\": %.2f,\n", g / timing.size());
        std::printf("  \"local_ms_per_frame\": %.2f,\n", l / timing.size());
        std::printf("  \"total_ms_per_frame\": %.2f,\n", (g + l) / timing.size());
    }
    if (total > 0) {
        std::printf("  \"teacher_match\": %ld,\n", match);
        std::printf("  \"teacher_total\": %ld,\n", total);
        std::printf("  \"teacher_flip_rate\": %.4f,\n", 1.0 - static_cast<double>(match) / total);
    }
    std::printf("  \"codes\": [\n");
    for (size_t i = 0; i < codes.size(); ++i) {
        std::printf("    [");
        for (int ch = 0; ch < kNVq; ++ch) {
            std::printf("%d%s", codes[i][ch + 1], ch + 1 < kNVq ? ", " : "");
        }
        std::printf("]%s\n", i + 1 < codes.size() ? "," : "");
    }
    std::printf("  ]\n}\n");

    if (!out_path.empty()) {
        std::ofstream of(out_path);
        for (const auto &row : codes) {
            for (int ch = 0; ch < kNVq; ++ch) of << row[ch + 1] << (ch + 1 < kNVq ? " " : "");
            of << "\n";
        }
    }
    return 0;
}
