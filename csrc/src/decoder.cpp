#include "moss/decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "moss/attn.h"
#include "moss/fp16.h"
#include "moss/parallel.h"
#include "moss/quant.h"

namespace moss {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kHeads = 4;
constexpr int kHeadDim = 64;
constexpr int kDModel = 256;
// 各段的环形窗口（帧）= context_duration × 该段帧率
constexpr int kContexts[4] = {500, 800, 1200, 1600};

}  // namespace

AudioDecoder::AudioDecoder(int threads) : par_(threads) {}

bool AudioDecoder::load(const std::string &path, std::string *err) {
    if (!wf_.load(path, err)) return false;

    q_out_w_ = wf_.matrix("q.out.w");
    q_out_b_ = wf_.vector("q.out.b");
    codebooks_.resize(16);
    for (int i = 0; i < 16; ++i) {
        Codebook &cb = codebooks_[i];
        cb.cb = wf_.matrix("q" + std::to_string(i) + ".cb");
        cb.out_w = wf_.matrix("q" + std::to_string(i) + ".out.w");
        cb.out_b = wf_.vector("q" + std::to_string(i) + ".out.b");
    }

    const int stage_ids[4] = {1, 3, 5, 7};
    const int patches[4] = {4, 2, 2, 2};
    const int layer_counts[4] = {4, 2, 2, 4};
    stages_.resize(4);
    for (int s = 0; s < 4; ++s) {
        const std::string p = "d" + std::to_string(stage_ids[s]) + ".";
        Stage &st = stages_[s];
        st.patch = patches[s];
        st.in_w = wf_.matrix(p + "in.w");
        st.out_w = wf_.matrix(p + "out.w");
        st.layers.resize(layer_counts[s]);
        for (int n = 0; n < layer_counts[s]; ++n) {
            const std::string lp = p + "l" + std::to_string(n) + ".";
            Layer &L = st.layers[n];
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
    info_ = "decoder weights=" + path + " bytes=" + std::to_string(wf_.bytes());
    return true;
}

void AudioDecoder::patch_forward(std::vector<float> &x, int &t, int &c, int p) const {
    if (p <= 1) return;
    const int d = c / p;
    std::vector<float> out(static_cast<size_t>(t) * p * d);
    for (int ti = 0; ti < t; ++ti) {
        const float *src = x.data() + static_cast<size_t>(ti) * c;
        for (int k = 0; k < p; ++k) {
            float *dst = out.data() + (static_cast<size_t>(ti) * p + k) * d;
            for (int ci = 0; ci < d; ++ci) dst[ci] = src[ci * p + k];
        }
    }
    x.swap(out);
    t *= p;
    c = d;
}

void AudioDecoder::run_layer(const Layer &layer, std::vector<float> &x, int t, int c) {
    audio_layer_forward(layer, x, t, c, current_window_, par_, pool_, buf_);
}

void AudioDecoder::run_stage(const Stage &stage, std::vector<float> &x, int &t, int &c) {
    std::vector<float> y(static_cast<size_t>(t) * kDModel);
    gemm_par(stage.in_w, x.data(), t, y.data(), pool_);
    for (const Layer &layer : stage.layers) run_layer(layer, y, t, kDModel);
    x.resize(static_cast<size_t>(t) * stage.out_w.rows);
    gemm_par(stage.out_w, y.data(), t, x.data(), pool_);
    c = stage.out_w.rows;
}


// ---------------- 流式 ----------------

void AudioDecoder::reset_stream() {
    stream_states_.clear();
    stream_states_.resize(stages_.size());
    // 各段的环形窗口：context_duration × 该段帧率（见 doc 附录 G）
    for (size_t s = 0; s < stages_.size(); ++s) {
        stream_states_[s].offset = 0;
        stream_states_[s].context = kContexts[s];
        stream_states_[s].layers.assign(stages_[s].layers.size(), LayerState{});
    }
}

void AudioDecoder::run_layer_stream(const Layer &layer, LayerState &state, int context, std::vector<float> &x, int t,
                                    int c, int offset) {
    Parallel &par = par_;
    const int rows = 3 * c;
    buf_.qkv.resize(static_cast<size_t>(t) * rows);
    buf_.ln.resize(static_cast<size_t>(t) * c);

    par.run(t, [&](int i) {
        audio_layernorm(x.data() + static_cast<size_t>(i) * c, layer.n1w, layer.n1b, c,
                  buf_.ln.data() + static_cast<size_t>(i) * c);
    });
    gemm_par(layer.attn_w, buf_.ln.data(), t, buf_.qkv.data(), pool_);
    par.run(t, [&](int i) {
        float *row = buf_.qkv.data() + static_cast<size_t>(i) * rows;
        audio_rope(row, offset + i, kHeads, kHeadDim);
        audio_rope(row + c, offset + i, kHeads, kHeadDim);
    });

    // keys = 缓存(最多 context 条) + 当前块；注意力走 AVX2 窗口内核
    const int n_keys = state.n + t;
    buf_.attn.resize(static_cast<size_t>(t) * c);
    const int workers = std::max(1, par.threads());
    const int per = (t + workers - 1) / workers;
    std::vector<std::vector<float>> bufs(static_cast<size_t>(workers));
    for (auto &b : bufs) b.assign(static_cast<size_t>(n_keys), 0.0f);

    par.run(workers, [&](int w) {
        const int i0 = w * per;
        const int i1 = std::min(t, i0 + per);
        if (i0 >= i1) return;
        attention_window(buf_.qkv.data(), t, i0, i1, c, kHeads, state.k.data(), state.v.data(), state.n,
                         context, buf_.attn.data(), bufs[static_cast<size_t>(w)].data());
    });

    // 追加当前块的 K/V，并裁剪到 context
    const size_t need = static_cast<size_t>(n_keys) * c;
    state.k.resize(need);
    state.v.resize(need);
    for (int i = 0; i < t; ++i) {
        const float *row = buf_.qkv.data() + static_cast<size_t>(i) * rows;
        std::memcpy(state.k.data() + static_cast<size_t>(state.n + i) * c, row + c, sizeof(float) * c);
        std::memcpy(state.v.data() + static_cast<size_t>(state.n + i) * c, row + 2 * c, sizeof(float) * c);
    }
    state.n = n_keys;
    if (state.n > context) {
        const int drop = state.n - context;
        std::memmove(state.k.data(), state.k.data() + static_cast<size_t>(drop) * c,
                     sizeof(float) * static_cast<size_t>(state.n - drop) * c);
        std::memmove(state.v.data(), state.v.data() + static_cast<size_t>(drop) * c,
                     sizeof(float) * static_cast<size_t>(state.n - drop) * c);
        state.n -= drop;
        state.k.resize(static_cast<size_t>(state.n) * c);
        state.v.resize(static_cast<size_t>(state.n) * c);
    }

    buf_.res.resize(static_cast<size_t>(t) * c);
    gemm_par(layer.proj_w, buf_.attn.data(), t, buf_.res.data(), pool_);
    par.run(t, [&](int i) {
        float *row = x.data() + static_cast<size_t>(i) * c;
        const float *r = buf_.res.data() + static_cast<size_t>(i) * c;
        for (int k = 0; k < c; ++k) row[k] += layer.ls1.data[k] * r[k];
    });

    const int ffn = layer.f1_w.rows;
    buf_.ffn.resize(static_cast<size_t>(t) * ffn);
    par.run(t, [&](int i) {
        audio_layernorm(x.data() + static_cast<size_t>(i) * c, layer.n2w, layer.n2b, c,
                  buf_.ln.data() + static_cast<size_t>(i) * c);
    });
    gemm_par(layer.f1_w, buf_.ln.data(), t, buf_.ffn.data(), pool_);
    par.run(t, [&](int i) {
        float *f = buf_.ffn.data() + static_cast<size_t>(i) * ffn;
        for (int k = 0; k < ffn; ++k) f[k] = audio_gelu(f[k]);
    });
    gemm_par(layer.f2_w, buf_.ffn.data(), t, buf_.res.data(), pool_);
    par.run(t, [&](int i) {
        float *row = x.data() + static_cast<size_t>(i) * c;
        const float *r = buf_.res.data() + static_cast<size_t>(i) * c;
        for (int k = 0; k < c; ++k) row[k] += layer.ls2.data[k] * r[k];
    });
}

void AudioDecoder::run_stage_stream(const Stage &stage, StageStream &state, std::vector<float> &x, int &t, int &c) {
    std::vector<float> y(static_cast<size_t>(t) * kDModel);
    gemm_par(stage.in_w, x.data(), t, y.data(), pool_);
    const int offset = state.offset;
    for (size_t n = 0; n < stage.layers.size(); ++n) {
        run_layer_stream(stage.layers[n], state.layers[n], state.context, y, t, kDModel, offset);
    }
    state.offset += t;
    x.resize(static_cast<size_t>(t) * stage.out_w.rows);
    gemm_par(stage.out_w, y.data(), t, x.data(), pool_);
    c = stage.out_w.rows;
}

std::vector<float> AudioDecoder::decode_chunk(const int32_t *codes, int frames) {
    if (stream_states_.empty()) reset_stream();
    int t = frames;
    std::vector<float> x = quantize_front(codes, frames, &t);
    int channels = q_out_w_.rows;

    for (size_t s = 0; s < stages_.size(); ++s) {
        patch_forward(x, t, channels, stages_[s].patch);
        run_stage_stream(stages_[s], stream_states_[s], x, t, channels);
    }
    patch_forward(x, t, channels, 240);
    std::vector<float> out(static_cast<size_t>(t));
    for (int i = 0; i < t; ++i) out[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
    return out;
}
std::vector<float> AudioDecoder::quantize_front(const int32_t *codes, int frames, int *out_t) {
    const int t = frames;
    // 16 路查表 + out_proj 累加 -> [T][512]
    std::vector<float> z(static_cast<size_t>(t) * 512, 0.0f);
    for (int i = 0; i < 16; ++i) {
        const Codebook &cb = codebooks_[i];
        for (int ti = 0; ti < t; ++ti) {
            const int code = codes[static_cast<size_t>(i) * t + ti];
            float emb[8];
            std::memcpy(emb, static_cast<const float *>(cb.cb.data) + static_cast<size_t>(code) * 8, sizeof(emb));
            float *zi = z.data() + static_cast<size_t>(ti) * 512;
            for (int r = 0; r < 512; ++r) {
                const float *w = static_cast<const float *>(cb.out_w.data) + static_cast<size_t>(r) * 8;
                float acc = cb.out_b.data[r];
                for (int k = 0; k < 8; ++k) acc += w[k] * emb[k];
                zi[r] += acc;
            }
        }
    }
    std::vector<float> x(static_cast<size_t>(t) * q_out_w_.rows);
    gemm_par(q_out_w_, z.data(), t, x.data(), pool_);
    for (int ti = 0; ti < t; ++ti) {
        float *row = x.data() + static_cast<size_t>(ti) * q_out_w_.rows;
        for (int k = 0; k < q_out_w_.rows; ++k) row[k] += q_out_b_.data[k];
    }
    *out_t = t;
    return x;
}
std::vector<float> AudioDecoder::decode(const int32_t *codes, int frames) {
    const auto started = Clock::now();
    int t = frames;

    std::vector<float> x = quantize_front(codes, frames, &t);
    int channels = q_out_w_.rows;

    stage_ms_.clear();
    int stage_index = 0;
    for (const Stage &stage : stages_) {
        patch_forward(x, t, channels, stage.patch);
        current_window_ = kContexts[stage_index++];
        const auto t0 = Clock::now();
        run_stage(stage, x, t, channels);
        stage_ms_.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    patch_forward(x, t, channels, 240);  // -> [T][1]

    // [samples][1] -> 交织 -> 去交织成立体声
    std::vector<float> inter(t);
    for (int i = 0; i < t; ++i) inter[i] = x[static_cast<size_t>(i)];
    std::vector<float> out(inter.size());
    std::memcpy(out.data(), inter.data(), inter.size() * sizeof(float));

    last_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    return out;
}

}  // namespace moss
