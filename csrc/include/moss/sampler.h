#pragma once

#include <cstdint>
#include <vector>

namespace moss {

struct SampleParams {
    bool greedy = true;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    // 结束判定只比较 audio_assistant_slot(9) 与 audio_end(7)（与上游候选集一致）。
    // 上游在这两个候选上按 softmax 概率抽，抽到 end 就停；概率虽小但每帧独立抽，长句几乎必然早停。
    // 默认贪心（end 严格更大才停），`end_greedy=false` 可还原上游的抽样行为。
    bool end_greedy = true;
    // 至少生成这么多帧才允许结束（防早停的安全阀，0 = 关闭）
    int min_frames = 0;
    // 前 N 帧强制贪心：第 0 帧没有音频上下文，分布最宽，采样容易抽出一个多余的元音起音
    int greedy_frames = 0;
};

// xoshiro128**：确定性、无外部依赖
struct Rng {
    uint32_t s[4] = {0x9E3779B9u, 0x243F6A88u, 0xB7E15162u, 0xDEADBEEFu};

    void seed(uint64_t v) {
        s[0] = static_cast<uint32_t>(v);
        s[1] = static_cast<uint32_t>(v >> 32) ^ 0x9E3779B9u;
        s[2] = 0x243F6A88u;
        s[3] = 0xB7E15162u;
        for (int i = 0; i < 8; ++i) next();
    }
    static uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
    uint32_t next() {
        const uint32_t result = rotl(s[1] * 5, 7) * 9;
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 11);
        return result;
    }
    float uniform() { return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f); }
};

// 从 logits 里选一个 token。prev 用于 repetition penalty（可为空）。
// scratch 是调用方复用缓冲，长度需 >= n。
int sample_token(const float *logits, int n, const SampleParams &params, const int *prev, int n_prev,
                 Rng &rng, std::vector<float> &scratch);

}  // namespace moss
