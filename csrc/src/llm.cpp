#include "moss/llm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "moss/fp16.h"
#include "moss/parallel.h"
#include "moss/quant.h"

namespace moss {
namespace {

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline float gelu_new(float v) {
    constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * v * (1.0f + std::tanh(k * (v + 0.044715f * v * v * v)));
}

// 是否该结束：只看 audio_end(7) 与 audio_assistant_slot(9) 两个候选（与上游候选集一致）。
// 上游在这两者上按 softmax 概率抽，抽到 end 就停 —— 单帧概率虽小，但每帧独立抽，
// 长句里几乎必然提前结束。默认改成贪心：end 严格更大才停。
inline bool end_reached(float l_slot, float l_end, const SampleParams &params, Rng &rng) {
    if (params.end_greedy) return l_end > l_slot;
    const double d = std::max(-80.0, std::min(80.0, static_cast<double>(l_end) - static_cast<double>(l_slot)));
    const float p_slot = static_cast<float>(1.0 / (1.0 + std::exp(d)));
    return rng.uniform() >= p_slot;
}

// 定义在文件下方（前缀 KV 一节）
uint64_t fp_weights(const std::string &path);

}  // namespace

LlmModel::LlmModel(int threads) : par_(threads) {}

void LlmModel::load_layer(Layer &layer, const std::string &p) {
    layer.ln1_w = wf_.vector(p + "ln1.w");
    layer.ln1_b = wf_.vector(p + "ln1.b");
    layer.ln2_w = wf_.vector(p + "ln2.w");
    layer.ln2_b = wf_.vector(p + "ln2.b");
    layer.attn_w = wf_.matrix(p + "attn.w");
    layer.attn_b = wf_.vector(p + "attn.b");
    layer.proj_w = wf_.matrix(p + "proj.w");
    layer.proj_b = wf_.vector(p + "proj.b");
    layer.fc_in_w = wf_.matrix(p + "fc_in.w");
    layer.fc_in_b = wf_.vector(p + "fc_in.b");
    layer.fc_out_w = wf_.matrix(p + "fc_out.w");
    layer.fc_out_b = wf_.vector(p + "fc_out.b");
}

bool LlmModel::load(const std::string &path, std::string *err) {
    if (!wf_.load(path, err)) return false;

    global_layers_.resize(12);
    for (int i = 0; i < 12; ++i) load_layer(global_layers_[i], "g" + std::to_string(i) + ".");
    load_layer(local_layer_, "local.");
    lnf_w_ = wf_.vector("lnf.w");
    lnf_b_ = wf_.vector("lnf.b");
    local_lnf_w_ = wf_.vector("local.lnf.w");
    local_lnf_b_ = wf_.vector("local.lnf.b");
    wte_ = wf_.matrix("wte");
    for (int ch = 0; ch < kNVq; ++ch) aemb_[ch] = wf_.matrix("aemb." + std::to_string(ch));

    g_k_.assign(static_cast<size_t>(12) * kMaxCtx * kHidden, 0.0f);
    g_v_.assign(static_cast<size_t>(12) * kMaxCtx * kHidden, 0.0f);
    l_k_.assign(static_cast<size_t>(kLocalLen) * kHidden, 0.0f);
    l_v_.assign(static_cast<size_t>(kLocalLen) * kHidden, 0.0f);

    x_.assign(kHidden, 0.0f);
    qkv_.assign(3 * kHidden, 0.0f);
    attn_buf_.assign(kHidden, 0.0f);
    mlp_buf_.assign(kFF, 0.0f);
    hidden_.assign(kHidden, 0.0f);
    ln_buf_.assign(kHidden, 0.0f);
    res_buf_.assign(kHidden, 0.0f);
    lx_.assign(kHidden, 0.0f);
    lqkv_.assign(3 * kHidden, 0.0f);
    lattn_.assign(kHidden, 0.0f);
    lmlp_.assign(kFF, 0.0f);
    logits_.assign(kVocab, 0.0f);

    cos_.assign(static_cast<size_t>(kMaxCtx) * (kHidden / 2), 0.0f);
    sin_.assign(static_cast<size_t>(kMaxCtx) * (kHidden / 2), 0.0f);
    const int half = kHeadDim / 2;
    for (int pos = 0; pos < kMaxCtx; ++pos) {
        for (int h = 0; h < kHeads; ++h) {
            for (int i = 0; i < half; ++i) {
                const double inv = 1.0 / std::pow(10000.0, static_cast<double>(2 * i) / kHeadDim);
                const double angle = static_cast<double>(pos) * inv;
                cos_[static_cast<size_t>(pos) * (kHidden / 2) + h * half + i] = static_cast<float>(std::cos(angle));
                sin_[static_cast<size_t>(pos) * (kHidden / 2) + h * half + i] = static_cast<float>(std::sin(angle));
            }
        }
    }

    pool_scratch_.bind(par_, kHidden);
    weights_fp_ = fp_weights(path);
    prefix_pos_ = 0;
    info_ = "weights=" + path + " bytes=" + std::to_string(wf_.bytes());
    reset();
    return true;
}

void LlmModel::reset() { pos_ = 0; }

void LlmModel::lookup_row(const Matrix &m, int row, float *out) const {
    const int cols = m.cols;
    switch (m.dtype) {
        case DType::F16: {
            const uint16_t *src = static_cast<const uint16_t *>(m.data) + static_cast<size_t>(row) * cols;
            for (int i = 0; i < cols; ++i) out[i] = half_to_float(src[i]);
            break;
        }
        case DType::F32: {
            const float *src = static_cast<const float *>(m.data) + static_cast<size_t>(row) * cols;
            std::memcpy(out, src, sizeof(float) * cols);
            break;
        }
        case DType::Q8_0:
            dequantize_row_q8_0(static_cast<const BlockQ8_0 *>(m.data) + static_cast<size_t>(row) * (cols / 32), out,
                                cols);
            break;
        case DType::Q4_0:
            dequantize_row_q4_0(static_cast<const BlockQ4_0 *>(m.data) + static_cast<size_t>(row) * (cols / 32), out,
                                cols);
            break;
        case DType::Q8_0_4X4:
            dequantize_row_q8_0_4x4(static_cast<const uint8_t *>(m.data) +
                                        static_cast<size_t>(row / 4) * (cols / 32) * 136,
                                    cols / 32, row % 4, out);
            break;
        case DType::Q4_0_4X4:
            dequantize_row_q4_0_4x4(static_cast<const uint8_t *>(m.data) +
                                        static_cast<size_t>(row / 4) * (cols / 32) * 72,
                                    cols / 32, row % 4, out);
            break;
    }
}

void LlmModel::embed_row(const Row &row, float *out) const {
    lookup_row(wte_, static_cast<int>(row[0]), out);
    std::array<float, kHidden> tmp{};
    for (int ch = 0; ch < kNVq; ++ch) {
        const int code = row[ch + 1];
        if (code == kAudioPad || code < 0 || code >= kCodebook) continue;
        lookup_row(aemb_[ch], code, tmp.data());
        for (int i = 0; i < kHidden; ++i) out[i] += tmp[i];
    }
}

void LlmModel::normalize(const float *x, const Vector &w, const Vector &b, float *out) const {
    float mean = 0.0f;
    for (int i = 0; i < kHidden; ++i) mean += x[i];
    mean /= kHidden;
    float var = 0.0f;
    for (int i = 0; i < kHidden; ++i) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= kHidden;
    const float inv = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < kHidden; ++i) out[i] = (x[i] - mean) * inv * w.data[i] + b.data[i];
}

void LlmModel::rope(float *vec, int position) const {
    const float *c = cos_.data() + static_cast<size_t>(position) * (kHidden / 2);
    const float *s = sin_.data() + static_cast<size_t>(position) * (kHidden / 2);
    for (int h = 0; h < kHeads; ++h) {
        float *base = vec + h * kHeadDim;
        for (int i = 0; i < kHeadDim / 2; ++i) {
            const float c0 = c[h * (kHeadDim / 2) + i];
            const float s0 = s[h * (kHeadDim / 2) + i];
            const float x0 = base[2 * i];
            const float x1 = base[2 * i + 1];
            base[2 * i] = x0 * c0 - x1 * s0;
            base[2 * i + 1] = x1 * c0 + x0 * s0;
        }
    }
}

void LlmModel::attention_core(const float *k_cache, const float *v_cache, int n_pos, const float *q, float *out,
                              float *scores) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (int h = 0; h < kHeads; ++h) {
        const float *qh = q + h * kHeadDim;
        float *oh = out + h * kHeadDim;
        float max_score = -1e30f;
        for (int p = 0; p < n_pos; ++p) {
            const float *kh = k_cache + static_cast<size_t>(p) * kHidden + h * kHeadDim;
            float dot = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) dot += qh[d] * kh[d];
            dot *= scale;
            scores[p] = dot;
            max_score = std::max(max_score, dot);
        }
        float sum = 0.0f;
        for (int p = 0; p < n_pos; ++p) {
            scores[p] = std::exp(scores[p] - max_score);
            sum += scores[p];
        }
        const float inv_sum = 1.0f / sum;
        for (int d = 0; d < kHeadDim; ++d) oh[d] = 0.0f;
        for (int p = 0; p < n_pos; ++p) {
            const float w = scores[p] * inv_sum;
            const float *vh = v_cache + static_cast<size_t>(p) * kHidden + h * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d) oh[d] += w * vh[d];
        }
    }
}

void LlmModel::attention(const Layer &layer, bool local, int layer_index, int position, const float *qkv,
                         float *out) {
    (void)layer;
    float *k_cache = local ? l_k_.data() : g_k_.data() + static_cast<size_t>(layer_index) * kMaxCtx * kHidden;
    float *v_cache = local ? l_v_.data() : g_v_.data() + static_cast<size_t>(layer_index) * kMaxCtx * kHidden;
    std::memcpy(k_cache + static_cast<size_t>(position) * kHidden, qkv + kHidden, sizeof(float) * kHidden);
    std::memcpy(v_cache + static_cast<size_t>(position) * kHidden, qkv + 2 * kHidden, sizeof(float) * kHidden);
    std::array<float, kMaxCtx> scores{};
    attention_core(k_cache, v_cache, position + 1, qkv, out, scores.data());
}

void LlmModel::block_forward(const Layer &layer, bool local, int layer_index, int position, float *x, float *qkv,
                             float *attn, float *mlp) {
    normalize(x, layer.ln1_w, layer.ln1_b, ln_buf_.data());
    gemv_par(layer.attn_w, ln_buf_.data(), qkv, pool_scratch_);
    for (int i = 0; i < 3 * kHidden; ++i) qkv[i] += layer.attn_b.data[i];
    rope(qkv, position);
    rope(qkv + kHidden, position);
    attention(layer, local, layer_index, position, qkv, attn);
    gemv_par(layer.proj_w, attn, res_buf_.data(), pool_scratch_);
    for (int i = 0; i < kHidden; ++i) x[i] += res_buf_[i] + layer.proj_b.data[i];

    normalize(x, layer.ln2_w, layer.ln2_b, ln_buf_.data());
    gemv_par(layer.fc_in_w, ln_buf_.data(), mlp, pool_scratch_);
    for (int i = 0; i < kFF; ++i) mlp[i] = gelu_new(mlp[i] + layer.fc_in_b.data[i]);
    gemv_par(layer.fc_out_w, mlp, res_buf_.data(), pool_scratch_);
    for (int i = 0; i < kHidden; ++i) x[i] += res_buf_[i] + layer.fc_out_b.data[i];
}

void LlmModel::global_step(const Row &row, float *hidden_out) {
    embed_row(row, x_.data());
    for (int i = 0; i < 12; ++i) {
        block_forward(global_layers_[i], false, i, pos_, x_.data(), qkv_.data(), attn_buf_.data(), mlp_buf_.data());
    }
    normalize(x_.data(), lnf_w_, lnf_b_, hidden_out);
    ++pos_;
}

void LlmModel::local_reset() { lpos_ = 0; }

void LlmModel::local_step(const float *input, float *out) {
    std::memcpy(lx_.data(), input, sizeof(float) * kHidden);
    block_forward(local_layer_, true, 0, lpos_, lx_.data(), lqkv_.data(), lattn_.data(), lmlp_.data());
    normalize(lx_.data(), local_lnf_w_, local_lnf_b_, out);
    ++lpos_;
}

void LlmModel::text_logits(const float *hidden, float *logits) { gemv_par(wte_, hidden, logits, pool_scratch_); }

void LlmModel::audio_logits(int channel, const float *hidden, float *logits) {
    gemv_par(aemb_[channel], hidden, logits, pool_scratch_);
}

// prefill：按层批量。权重只读一遍，attention 用已写入的 KV cache。
// start>0 表示接着已有的前缀 KV 往后写（多段文本 / 前缀缓存复用）。
void LlmModel::prefill(const std::vector<Row> &rows, int start) {
    start = std::max(0, std::min(start, kMaxCtx));
    const int m = static_cast<int>(std::min<size_t>(rows.size(), static_cast<size_t>(kMaxCtx - start)));
    if (m <= 0) return;
    Parallel &par = par_;

    b_x_.resize(static_cast<size_t>(m) * kHidden);
    b_ln_.resize(static_cast<size_t>(m) * kHidden);
    b_qkv_.resize(static_cast<size_t>(m) * 3 * kHidden);
    b_att_.resize(static_cast<size_t>(m) * kHidden);
    b_mlp_.resize(static_cast<size_t>(m) * kFF);
    b_res_.resize(static_cast<size_t>(m) * kHidden);
    b_scores_.resize(static_cast<size_t>(m) * kMaxCtx);

    for (int i = 0; i < m; ++i) embed_row(rows[i], b_x_.data() + static_cast<size_t>(i) * kHidden);

    for (int layer = 0; layer < 12; ++layer) {
        const Layer &L = global_layers_[layer];
        float *k_cache = g_k_.data() + static_cast<size_t>(layer) * kMaxCtx * kHidden;
        float *v_cache = g_v_.data() + static_cast<size_t>(layer) * kMaxCtx * kHidden;

        par.run(m, [&](int i) {
            normalize(b_x_.data() + static_cast<size_t>(i) * kHidden, L.ln1_w, L.ln1_b,
                      b_ln_.data() + static_cast<size_t>(i) * kHidden);
        });
        gemm_par(L.attn_w, b_ln_.data(), m, b_qkv_.data(), pool_scratch_);

        par.run(m, [&](int i) {
            float *qkv = b_qkv_.data() + static_cast<size_t>(i) * 3 * kHidden;
            const int p = start + i;
            for (int k = 0; k < 3 * kHidden; ++k) qkv[k] += L.attn_b.data[k];
            rope(qkv, p);
            rope(qkv + kHidden, p);
            std::memcpy(k_cache + static_cast<size_t>(p) * kHidden, qkv + kHidden, sizeof(float) * kHidden);
            std::memcpy(v_cache + static_cast<size_t>(p) * kHidden, qkv + 2 * kHidden, sizeof(float) * kHidden);
        });

        par.run(m, [&](int i) {
            attention_core(k_cache, v_cache, start + i + 1, b_qkv_.data() + static_cast<size_t>(i) * 3 * kHidden,
                           b_att_.data() + static_cast<size_t>(i) * kHidden,
                           b_scores_.data() + static_cast<size_t>(i) * kMaxCtx);
        });

        gemm_par(L.proj_w, b_att_.data(), m, b_res_.data(), pool_scratch_);
        par.run(m, [&](int i) {
            float *x = b_x_.data() + static_cast<size_t>(i) * kHidden;
            const float *r = b_res_.data() + static_cast<size_t>(i) * kHidden;
            for (int k = 0; k < kHidden; ++k) x[k] += r[k] + L.proj_b.data[k];
        });

        par.run(m, [&](int i) {
            normalize(b_x_.data() + static_cast<size_t>(i) * kHidden, L.ln2_w, L.ln2_b,
                      b_ln_.data() + static_cast<size_t>(i) * kHidden);
        });
        gemm_par(L.fc_in_w, b_ln_.data(), m, b_mlp_.data(), pool_scratch_);
        par.run(m, [&](int i) {
            float *mlp = b_mlp_.data() + static_cast<size_t>(i) * kFF;
            for (int k = 0; k < kFF; ++k) mlp[k] = gelu_new(mlp[k] + L.fc_in_b.data[k]);
        });
        gemm_par(L.fc_out_w, b_mlp_.data(), m, b_res_.data(), pool_scratch_);
        par.run(m, [&](int i) {
            float *x = b_x_.data() + static_cast<size_t>(i) * kHidden;
            const float *r = b_res_.data() + static_cast<size_t>(i) * kHidden;
            for (int k = 0; k < kHidden; ++k) x[k] += r[k] + L.fc_out_b.data[k];
        });
    }

    par.run(m, [&](int i) {
        normalize(b_x_.data() + static_cast<size_t>(i) * kHidden, lnf_w_, lnf_b_,
                  b_ln_.data() + static_cast<size_t>(i) * kHidden);
    });
    std::memcpy(hidden_.data(), b_ln_.data() + static_cast<size_t>(m - 1) * kHidden, sizeof(float) * kHidden);
    pos_ = start + m;
}

// ---------------- 前缀 KV ----------------

namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

inline uint64_t fnv_bytes(uint64_t h, const void *data, size_t n) {
    const uint8_t *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}

inline uint64_t fp_rows(const std::vector<Row> &rows) {
    uint64_t h = kFnvOffset;
    for (const Row &r : rows) h = fnv_bytes(h, r.data(), sizeof(int32_t) * r.size());
    return h;
}

// 权重指纹：文件大小 + 首 4 KB。够用来发现「换了权重还用旧缓存」。
uint64_t fp_weights(const std::string &path) {
    uint64_t h = kFnvOffset;
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    uint8_t head[4096];
    const size_t got = std::fread(head, 1, sizeof(head), f);
    std::fclose(f);
    const int64_t sz = size;
    h = fnv_bytes(h, &sz, sizeof(sz));
    return fnv_bytes(h, head, got);
}

constexpr char kKvMagic[8] = {'M', 'O', 'S', 'S', 'P', 'F', 'X', '1'};

struct KvHeader {
    char magic[8];
    uint32_t version = 1;
    uint32_t layers = 0;
    uint32_t n_pos = 0;
    uint32_t hidden = 0;
    uint64_t rows_fp = 0;
    uint64_t weights_fp = 0;
};

}  // namespace

void LlmModel::snapshot_prefix() {
    prefix_pos_ = pos_;
    const size_t per_layer = static_cast<size_t>(prefix_pos_) * kHidden;
    p_k_.resize(static_cast<size_t>(12) * per_layer);
    p_v_.resize(static_cast<size_t>(12) * per_layer);
    for (int layer = 0; layer < 12; ++layer) {
        const size_t src = static_cast<size_t>(layer) * kMaxCtx * kHidden;
        std::memcpy(p_k_.data() + static_cast<size_t>(layer) * per_layer, g_k_.data() + src,
                    sizeof(float) * per_layer);
        std::memcpy(p_v_.data() + static_cast<size_t>(layer) * per_layer, g_v_.data() + src,
                    sizeof(float) * per_layer);
    }
}

void LlmModel::restore_prefix() {
    if (prefix_pos_ <= 0) {
        pos_ = 0;
        return;
    }
    const size_t per_layer = static_cast<size_t>(prefix_pos_) * kHidden;
    for (int layer = 0; layer < 12; ++layer) {
        const size_t dst = static_cast<size_t>(layer) * kMaxCtx * kHidden;
        std::memcpy(g_k_.data() + dst, p_k_.data() + static_cast<size_t>(layer) * per_layer,
                    sizeof(float) * per_layer);
        std::memcpy(g_v_.data() + dst, p_v_.data() + static_cast<size_t>(layer) * per_layer,
                    sizeof(float) * per_layer);
    }
    pos_ = prefix_pos_;
}

bool LlmModel::save_prefix(const std::string &path, const std::vector<Row> &rows, std::string *err) {
    if (pos_ <= 0) {
        if (err) *err = "前缀为空，无法保存";
        return false;
    }
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        if (err) *err = "写不了前缀缓存: " + path;
        return false;
    }
    KvHeader h{};
    std::memcpy(h.magic, kKvMagic, 8);
    h.layers = 12;
    h.n_pos = static_cast<uint32_t>(pos_);
    h.hidden = kHidden;
    h.rows_fp = fp_rows(rows);
    h.weights_fp = weights_fp_;
    std::fwrite(&h, sizeof(h), 1, f);
    for (int layer = 0; layer < 12; ++layer) {
        const size_t src = static_cast<size_t>(layer) * kMaxCtx * kHidden;
        std::fwrite(g_k_.data() + src, sizeof(float), static_cast<size_t>(pos_) * kHidden, f);
        std::fwrite(g_v_.data() + src, sizeof(float), static_cast<size_t>(pos_) * kHidden, f);
    }
    std::fclose(f);
    return true;
}

bool LlmModel::load_prefix(const std::string &path, const std::vector<Row> &rows, std::string *err) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "打不开前缀缓存: " + path;
        return false;
    }
    KvHeader h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1 || std::memcmp(h.magic, kKvMagic, 8) != 0 ||
        h.version != 1 || h.layers != 12 || h.hidden != static_cast<uint32_t>(kHidden) ||
        h.n_pos == 0 || h.n_pos > static_cast<uint32_t>(kMaxCtx)) {
        std::fclose(f);
        if (err) *err = "前缀缓存格式不对: " + path;
        return false;
    }
    if (h.rows_fp != fp_rows(rows)) {
        std::fclose(f);
        if (err) *err = "前缀缓存与 prompt 不匹配（rows 指纹不符）";
        return false;
    }
    if (h.weights_fp != weights_fp_) {
        std::fclose(f);
        if (err) *err = "前缀缓存与当前权重不匹配（weights 指纹不符）";
        return false;
    }
    for (int layer = 0; layer < 12; ++layer) {
        const size_t dst = static_cast<size_t>(layer) * kMaxCtx * kHidden;
        const size_t n = static_cast<size_t>(h.n_pos) * kHidden;
        if (std::fread(g_k_.data() + dst, sizeof(float), n, f) != n ||
            std::fread(g_v_.data() + dst, sizeof(float), n, f) != n) {
            std::fclose(f);
            if (err) *err = "前缀缓存被截断: " + path;
            return false;
        }
    }
    std::fclose(f);
    pos_ = static_cast<int>(h.n_pos);
    snapshot_prefix();  // 同时留一份在内存里，供多段文本反复复位
    return true;
}

std::vector<Row> LlmModel::generate(int max_frames, const SampleParams &audio_params, uint64_t seed,
                                    std::vector<FrameTiming> *timing, const std::vector<Row> *forced,
                                    const std::function<void(int, const Row &)> &on_frame) {
    std::vector<Row> frames;
    Rng rng;
    rng.seed(seed);
    std::array<std::vector<int>, kNVq> history;
    gate_dump_.clear();

    for (int f = 0; f < max_frames; ++f) {
        if (pos_ >= kMaxCtx) break;
        const auto t_local0 = Clock::now();

        local_reset();
        local_step(hidden_.data(), lx_.data());
        text_logits(lx_.data(), logits_.data());
        const float l_slot = logits_[kSlotToken];
        const float l_end = logits_[kEndToken];
        if (!gate_dump_path_.empty()) {
            int argmax_id = 0;
            float argmax_v = logits_[0];
            for (int v = 1; v < kVocab; ++v) {
                if (logits_[v] > argmax_v) {
                    argmax_v = logits_[v];
                    argmax_id = v;
                }
            }
            gate_dump_.push_back(static_cast<float>(f));
            gate_dump_.push_back(static_cast<float>(argmax_id));
            gate_dump_.push_back(argmax_v - l_slot);
            gate_dump_.push_back(l_slot - l_end);
        }
        if (f >= audio_params.min_frames && end_reached(l_slot, l_end, audio_params, rng)) break;

        Row frame{};
        std::array<float, kHidden> input{};
        lookup_row(wte_, kSlotToken, input.data());
        // 开头几帧走贪心：首帧没有上下文，采样抽到通用元音就会多出一声
        SampleParams frame_params = audio_params;
        if (f < audio_params.greedy_frames) frame_params.greedy = true;
        for (int ch = 0; ch < kNVq; ++ch) {
            local_step(input.data(), lx_.data());
            audio_logits(ch, lx_.data(), logits_.data());
            if (!logits_dump_path_.empty()) {
                logits_dump_.insert(logits_dump_.end(), logits_.data(), logits_.data() + kCodebook);
            }
            const int predicted = sample_token(logits_.data(), kCodebook, frame_params, history[ch].data(),
                                               static_cast<int>(history[ch].size()), rng, sample_scratch_);
            const int used = forced ? (*forced)[f][ch + 1] : predicted;
            frame[ch + 1] = predicted;
            history[ch].push_back(used);
            lookup_row(aemb_[ch], used, input.data());
        }
        const double local_ms = ms_since(t_local0);
        frames.push_back(frame);
        if (on_frame) on_frame(f, frame);

        if (f + 1 < max_frames && pos_ < kMaxCtx) {
            Row next{};
            next[0] = kSlotToken;
            for (int ch = 0; ch < kNVq; ++ch) next[ch + 1] = forced ? (*forced)[f][ch + 1] : frame[ch + 1];
            const auto t_global0 = Clock::now();
            global_step(next, hidden_.data());
            if (timing) timing->push_back(FrameTiming{ms_since(t_global0), local_ms});
        } else if (timing) {
            timing->push_back(FrameTiming{0.0, local_ms});
        }
    }
    if (!logits_dump_path_.empty()) {
        std::FILE *fp = std::fopen(logits_dump_path_.c_str(), "wb");
        if (fp) {
            std::fwrite(logits_dump_.data(), sizeof(float), logits_dump_.size(), fp);
            std::fclose(fp);
        }
    }
    if (!gate_dump_path_.empty()) {
        std::FILE *fp = std::fopen(gate_dump_path_.c_str(), "wb");
        if (fp) {
            std::fprintf(fp, "frame,argmax_id,argmax_minus_slot,slot_minus_end\n");
            for (size_t i = 0; i + 3 < gate_dump_.size(); i += 4) {
                std::fprintf(fp, "%d,%d,%.4f,%.4f\n", static_cast<int>(gate_dump_[i]),
                             static_cast<int>(gate_dump_[i + 1]), gate_dump_[i + 2], gate_dump_[i + 3]);
            }
            std::fclose(fp);
        }
    }
    return frames;
}

}  // namespace moss
