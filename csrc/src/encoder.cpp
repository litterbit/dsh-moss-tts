#include "moss/encoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "moss/fp16.h"

namespace moss {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kDModel = 256;
constexpr int kCodeDim = 512;
// 各段的因果窗口（token 数）= context_duration ÷ 该段帧间隔，与解码器正好反序
constexpr int kWindows[4] = {1600, 1200, 800, 500};
constexpr int kPatch = 240;   // 输入级 patch
constexpr int kTailPatch = 4; // 进量化器前的 patch
constexpr int kSamplePerFrame = 3840;  // 每帧的单声道样本数（80 ms）

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline void normalize_l2(float *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += static_cast<double>(v[i]) * v[i];
    const double norm = std::sqrt(sum);
    const float inv = static_cast<float>(1.0 / std::max(norm, 1e-12));
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

}  // namespace

AudioEncoder::AudioEncoder(int threads) : par_(threads) {}

bool AudioEncoder::load(const std::string &path, std::string *err) {
    if (!wf_.load(path, err)) return false;

    q_in_w_ = wf_.matrix("q.in.w");
    q_in_b_ = wf_.vector("q.in.b");

    lfqs_.resize(16);
    for (int i = 0; i < 16; ++i) {
        Lfq &q = lfqs_[i];
        q.in_w = wf_.matrix("q" + std::to_string(i) + ".in.w");
        q.in_b = wf_.vector("q" + std::to_string(i) + ".in.b");
        q.out_w = wf_.matrix("q" + std::to_string(i) + ".out.w");
        q.out_b = wf_.vector("q" + std::to_string(i) + ".out.b");
        const Matrix &cb = wf_.matrix("q" + std::to_string(i) + ".cb");
        if (cb.rows != 1024 || cb.cols != 8) {
            if (err) *err = "码本形状不对（期望 1024x8）";
            return false;
        }
        // LFQ 的最近邻是在**归一化**空间里算的：码本先归一化一次，之后每帧只需内积
        std::vector<float> raw(static_cast<size_t>(cb.rows) * cb.cols);
        if (cb.dtype == DType::F32) {
            std::memcpy(raw.data(), cb.data, sizeof(float) * raw.size());
        } else if (cb.dtype == DType::F16) {
            const uint16_t *src = static_cast<const uint16_t *>(cb.data);
            for (size_t k = 0; k < raw.size(); ++k) raw[k] = half_to_float(src[k]);
        } else {
            if (err) *err = "码本只支持 f16/f32";
            return false;
        }
        q.cb_raw = raw;
        q.cb_norm = raw;
        for (int r = 0; r < cb.rows; ++r) normalize_l2(q.cb_norm.data() + static_cast<size_t>(r) * 8, 8);
    }

    const int stage_ids[4] = {1, 3, 5, 7};
    const int patches[4] = {kPatch, 2, 2, 2};
    const int layer_counts[4] = {4, 2, 2, 4};
    stages_.resize(4);
    for (int s = 0; s < 4; ++s) {
        const std::string p = "e" + std::to_string(stage_ids[s]) + ".";
        Stage &st = stages_[s];
        st.patch = patches[s];
        st.window = kWindows[s];
        st.in_w = wf_.matrix(p + "in.w");
        st.out_w = wf_.matrix(p + "out.w");
        st.layers.resize(layer_counts[s]);
        for (int n = 0; n < layer_counts[s]; ++n) {
            const std::string lp = p + "l" + std::to_string(n) + ".";
            AudioLayer &L = st.layers[n];
            L.attn_w = wf_.matrix(lp + "attn.w");
            L.proj_w = wf_.matrix(lp + "proj.w");
            L.f1_w = wf_.matrix(lp + "f1.w");
            L.f2_w = wf_.matrix(lp + "f2.w");
            L.n1w = wf_.vector(lp + "n1.w");
            L.n1b = wf_.vector(lp + "n1.b");
            L.n2w = wf_.vector(lp + "n2.w");
            L.n2b = wf_.vector(lp + "n2.b");
            L.ls1 = wf_.vector(lp + "ls1");
            L.ls2 = wf_.vector(lp + "ls2");
        }
    }

    pool_.bind(par_, kDModel);
    info_ = "encoder weights=" + path + " bytes=" + std::to_string(wf_.bytes());
    return true;
}

void AudioEncoder::patch_down(std::vector<float> &x, int &t, int &c, int p) const {
    if (p <= 1) return;
    const int d = c * p;
    const int nt = t / p;
    std::vector<float> out(static_cast<size_t>(nt) * d);
    for (int ti = 0; ti < nt; ++ti) {
        float *dst = out.data() + static_cast<size_t>(ti) * d;
        for (int k = 0; k < p; ++k) {
            const float *src = x.data() + static_cast<size_t>(ti * p + k) * c;
            for (int ci = 0; ci < c; ++ci) dst[ci * p + k] = src[ci];
        }
    }
    x.swap(out);
    t = nt;
    c = d;
}

void AudioEncoder::run_stage(const Stage &stage, std::vector<float> &x, int &t, int &c) {
    patch_down(x, t, c, stage.patch);
    std::vector<float> y(static_cast<size_t>(t) * kDModel);
    gemm_par(stage.in_w, x.data(), t, y.data(), pool_);
    for (const AudioLayer &layer : stage.layers) {
        audio_layer_forward(layer, y, t, kDModel, stage.window, par_, pool_, buf_);
    }
    x.resize(static_cast<size_t>(t) * stage.out_w.rows);
    gemm_par(stage.out_w, y.data(), t, x.data(), pool_);
    c = stage.out_w.rows;
}

void AudioEncoder::run_quantizer(std::vector<float> &x, int t, std::vector<int32_t> *codes) {
    // z: [t][768] -> [t][512]
    std::vector<float> z(static_cast<size_t>(t) * kCodeDim);
    gemm_par(q_in_w_, x.data(), t, z.data(), pool_);
    for (int i = 0; i < t; ++i) {
        float *row = z.data() + static_cast<size_t>(i) * kCodeDim;
        for (int k = 0; k < kCodeDim; ++k) row[k] += q_in_b_.data[k];
    }

    codes->assign(static_cast<size_t>(t) * 16, 0);
    std::vector<float> residual = z;
    std::vector<float> low(static_cast<size_t>(t) * 8);
    std::vector<float> gathered(static_cast<size_t>(t) * 8);
    std::vector<float> rec(static_cast<size_t>(t) * kCodeDim);

    for (int qi = 0; qi < 16; ++qi) {
        const Lfq &q = lfqs_[qi];
        // 512 -> 8（每路的 in_proj），然后 L2 归一化
        gemm_par(q.in_w, residual.data(), t, low.data(), pool_);
        for (int i = 0; i < t; ++i) {
            float *row = low.data() + static_cast<size_t>(i) * 8;
            for (int k = 0; k < 8; ++k) row[k] += q.in_b.data[k];
            normalize_l2(row, 8);
        }
        // 与归一化码本内积取 argmax（距离 = 2 - 2·dot）
        const float *cb = q.cb_norm.data();
        par_.run(t, [&](int i) {
            const float *row = low.data() + static_cast<size_t>(i) * 8;
            float best = -1e30f;
            int best_id = 0;
            for (int k = 0; k < 1024; ++k) {
                const float *c = cb + static_cast<size_t>(k) * 8;
                const float dot = row[0] * c[0] + row[1] * c[1] + row[2] * c[2] + row[3] * c[3] +
                                  row[4] * c[4] + row[5] * c[5] + row[6] * c[6] + row[7] * c[7];
                if (dot > best) {
                    best = dot;
                    best_id = k;
                }
            }
            (*codes)[static_cast<size_t>(i) * 16 + qi] = best_id;
        });
        // 用被选中的码本向量重建 512 维，扣掉残差
        for (int i = 0; i < t; ++i) {
            const int id = (*codes)[static_cast<size_t>(i) * 16 + qi];
            std::memcpy(gathered.data() + static_cast<size_t>(i) * 8,
                        q.cb_raw.data() + static_cast<size_t>(id) * 8, sizeof(float) * 8);
        }
        gemm_par(q.out_w, gathered.data(), t, rec.data(), pool_);
        for (int i = 0; i < t; ++i) {
            float *res = residual.data() + static_cast<size_t>(i) * kCodeDim;
            const float *r = rec.data() + static_cast<size_t>(i) * kCodeDim;
            for (int k = 0; k < kCodeDim; ++k) res[k] -= r[k] + q.out_b.data[k];
        }
    }
}

int AudioEncoder::encode(const float *pcm, int per_channel, std::vector<int32_t> *codes) {
    const auto t_all = Clock::now();
    stage_ms_.clear();
    if (per_channel <= 0) return 0;

    // 1) 交织样本补零到 7680 的倍数（每帧 3840 单声道样本 = 7680 交织样本）
    const int interleaved = per_channel * 2;
    const int padded = ((interleaved + 7679) / 7680) * 7680;
    std::vector<float> x(static_cast<size_t>(padded), 0.0f);
    std::memcpy(x.data(), pcm, sizeof(float) * static_cast<size_t>(interleaved));

    int t = padded;
    int c = 1;
    for (size_t s = 0; s < stages_.size(); ++s) {
        const auto t0 = Clock::now();
        run_stage(stages_[s], x, t, c);
        stage_ms_.push_back(ms_since(t0));
    }
    patch_down(x, t, c, kTailPatch);

    const auto tq = Clock::now();
    run_quantizer(x, t, codes);
    stage_ms_.push_back(ms_since(tq));

    // 2) 只保留完整帧（上游用 floor(len/3840)，尾部的补零帧丢弃）
    const int valid = std::min(t, per_channel / kSamplePerFrame);
    codes->resize(static_cast<size_t>(valid) * 16);
    last_ms_ = ms_since(t_all);
    return valid;
}

}  // namespace moss
