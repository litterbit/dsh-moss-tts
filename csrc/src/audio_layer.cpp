#include "moss/audio_layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "moss/attn.h"

namespace moss {
namespace {

constexpr int kHeads = 4;
constexpr int kHeadDim = 64;

}  // namespace

void audio_layernorm(const float *x, const Vector &w, const Vector &b, int n, float *out) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= static_cast<float>(n);
    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= static_cast<float>(n);
    const float inv = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < n; ++i) out[i] = (x[i] - mean) * inv * w.data[i] + b.data[i];
}

void audio_rope(float *vec, int pos, int heads, int head_dim, float max_period) {
    const int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        float *base = vec + h * head_dim;
        for (int i = 0; i < half; ++i) {
            const float freq = std::exp(-std::log(max_period) * 2.0f * static_cast<float>(i) / head_dim);
            const float ang = static_cast<float>(pos) * freq;
            const float c = std::cos(ang), s = std::sin(ang);
            const float x0 = base[2 * i], x1 = base[2 * i + 1];
            base[2 * i] = x0 * c - x1 * s;
            base[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

float audio_gelu(float v) { return 0.5f * v * (1.0f + std::erff(v * 0.7071067811865476f)); }

void audio_layer_forward(const AudioLayer &layer, std::vector<float> &x, int t, int c, int window,
                         Parallel &par, GemvPoolScratch &pool, AudioLayerBuffers &buf) {
    const int rows = 3 * c;
    buf.qkv.resize(static_cast<size_t>(t) * rows);
    buf.ln.resize(static_cast<size_t>(t) * c);

    // LN1 → QKV
    par.run(t, [&](int i) {
        audio_layernorm(x.data() + static_cast<size_t>(i) * c, layer.n1w, layer.n1b, c,
                        buf.ln.data() + static_cast<size_t>(i) * c);
    });
    gemm_par(layer.attn_w, buf.ln.data(), t, buf.qkv.data(), pool);

    // RoPE on q,k
    par.run(t, [&](int i) {
        float *row = buf.qkv.data() + static_cast<size_t>(i) * rows;
        audio_rope(row, i, kHeads, kHeadDim);
        audio_rope(row + c, i, kHeads, kHeadDim);
    });

    // 因果注意力：按工作线程切查询区间，每线程一块 scores 缓冲
    buf.attn.resize(static_cast<size_t>(t) * c);
    {
        const int workers = std::max(1, par.threads());
        const int per = (t + workers - 1) / workers;
        std::vector<std::vector<float>> scores(static_cast<size_t>(workers));
        for (auto &s : scores) s.assign(static_cast<size_t>(t), 0.0f);
        par.run(workers, [&](int w) {
            const int i0 = w * per;
            const int i1 = std::min(t, i0 + per);
            if (i0 >= i1) return;
            attention_causal(buf.qkv.data(), t, i0, i1, c, kHeads, window, buf.attn.data(),
                             scores[static_cast<size_t>(w)].data());
        });
    }

    buf.res.resize(static_cast<size_t>(t) * c);
    gemm_par(layer.proj_w, buf.attn.data(), t, buf.res.data(), pool);
    par.run(t, [&](int i) {
        float *row = x.data() + static_cast<size_t>(i) * c;
        const float *r = buf.res.data() + static_cast<size_t>(i) * c;
        for (int k = 0; k < c; ++k) row[k] += layer.ls1.data[k] * r[k];
    });

    const int ffn = layer.f1_w.rows;
    buf.ffn.resize(static_cast<size_t>(t) * ffn);
    par.run(t, [&](int i) {
        audio_layernorm(x.data() + static_cast<size_t>(i) * c, layer.n2w, layer.n2b, c,
                        buf.ln.data() + static_cast<size_t>(i) * c);
    });
    gemm_par(layer.f1_w, buf.ln.data(), t, buf.ffn.data(), pool);
    par.run(t, [&](int i) {
        float *f = buf.ffn.data() + static_cast<size_t>(i) * ffn;
        for (int k = 0; k < ffn; ++k) f[k] = audio_gelu(f[k]);
    });
    gemm_par(layer.f2_w, buf.ffn.data(), t, buf.res.data(), pool);
    par.run(t, [&](int i) {
        float *row = x.data() + static_cast<size_t>(i) * c;
        const float *r = buf.res.data() + static_cast<size_t>(i) * c;
        for (int k = 0; k < c; ++k) row[k] += layer.ls2.data[k] * r[k];
    });
}

}  // namespace moss
