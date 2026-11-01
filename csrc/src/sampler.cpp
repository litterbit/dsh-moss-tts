#include "moss/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace moss {

int sample_token(const float *logits, int n, const SampleParams &params, const int *prev, int n_prev,
                 Rng &rng, std::vector<float> &scratch) {
    scratch.assign(logits, logits + n);

    // repetition penalty：只惩罚历史里出现过的 token
    if (params.repetition_penalty != 1.0f && prev != nullptr && n_prev > 0) {
        for (int i = 0; i < n_prev; ++i) {
            const int t = prev[i];
            if (t < 0 || t >= n) continue;
            float &v = scratch[t];
            v = v < 0.0f ? v * params.repetition_penalty : v / params.repetition_penalty;
        }
    }

    if (params.greedy || params.top_k == 1) {
        int best = 0;
        for (int i = 1; i < n; ++i)
            if (scratch[i] > scratch[best]) best = i;
        return best;
    }

    if (params.temperature > 0.0f && params.temperature != 1.0f) {
        const float inv = 1.0f / params.temperature;
        for (int i = 0; i < n; ++i) scratch[i] *= inv;
    }

    // top-k：部分排序找出前 k 个
    int k = params.top_k > 0 ? std::min(params.top_k, n) : n;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (k < n) {
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return scratch[a] > scratch[b]; });
        idx.resize(k);
    } else {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return scratch[a] > scratch[b]; });
    }

    // softmax（只在候选集上）
    float maxv = scratch[idx[0]];
    float sum = 0.0f;
    for (int i : idx) {
        scratch[i] = std::exp(scratch[i] - maxv);
        sum += scratch[i];
    }

    // top-p：累积到 p 为止
    if (params.top_p < 1.0f) {
        float cum = 0.0f;
        size_t keep = 0;
        for (; keep < idx.size(); ++keep) {
            cum += scratch[idx[keep]] / sum;
            if (cum >= params.top_p) {
                ++keep;
                break;
            }
        }
        idx.resize(std::max<size_t>(1, keep));
        sum = 0.0f;
        for (int i : idx) sum += scratch[i];
    }

    float r = rng.uniform() * sum;
    for (int i : idx) {
        r -= scratch[i];
        if (r <= 0.0f) return i;
    }
    return idx.back();
}

}  // namespace moss
