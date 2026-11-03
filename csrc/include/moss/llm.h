#pragma once

#include <array>
#include <functional>
#include <cstdint>
#include <string>
#include <vector>

#include "moss/common.h"
#include "moss/gemv.h"
#include "moss/parallel.h"
#include "moss/sampler.h"
#include "moss/weights.h"

namespace moss {

constexpr int kHidden = 768;
constexpr int kHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kFF = 3072;
constexpr int kVocab = 16384;
constexpr int kNVq = 16;
constexpr int kCodebook = 1024;
constexpr int kAudioPad = 1024;
constexpr int kSlotToken = 9;
constexpr int kEndToken = 7;
constexpr int kMaxCtx = 768;
constexpr int kLocalLen = 17;

// 一行输入：[text_id, code0..code15]
using Row = std::array<int32_t, kNVq + 1>;

struct FrameTiming {
    double global_ms = 0.0;
    double local_ms = 0.0;
};

class LlmModel {
public:
    // threads：本引擎独占的线程池大小（不与解码器共享）
    explicit LlmModel(int threads = 1);

    bool load(const std::string &weights_path, std::string *err);

    // 清空 KV 与位置
    void reset();

    // 把 prompt 行喂进 global KV。start>0 时接着已有 KV 往后写（RoPE 与注意力都从 start 起算）。
    // 结束后 hidden() 是「最后一行」的 ln_f 输出，pos_ = start + rows.size()。
    void prefill(const std::vector<Row> &rows, int start = 0);

    // ---- 前缀 KV（音色/克隆那一段只算一次）----
    // 把当前 [0,pos_) 的 KV 记为「可复原的前缀」
    void snapshot_prefix();
    // 把 KV 复位到前缀末尾（多段文本时每段开头调一次）
    void restore_prefix();
    int prefix_pos() const { return prefix_pos_; }

    // 前缀缓存落盘/读盘；rows 是这段前缀本身，用来做指纹校验（防错配）
    bool save_prefix(const std::string &path, const std::vector<Row> &rows, std::string *err);
    bool load_prefix(const std::string &path, const std::vector<Row> &rows, std::string *err);

    // 调试：把「prefill 完的状态」（KV + 最后一行 hidden）整份存/取。
    // 用来隔离 prefill 与 decode 各自的精度影响（f16 prefill → int8 decode）。
    bool save_state(const std::string &path, const std::vector<Row> &rows, std::string *err);
    bool load_state(const std::string &path, const std::vector<Row> &rows, bool ignore_weights_fp,
                    std::string *err);

    // 生成若干帧；返回每帧 16 个 code
    // forced 非空时为 teacher forcing：输入用 forced 的 codes，输出的是模型自己的预测，
    // 用于逐帧比较数值误差（自由生成会因 argmax 翻转级联，不适合做精度指标）。
    std::vector<Row> generate(int max_frames, const SampleParams &audio_params, uint64_t seed,
                              std::vector<FrameTiming> *timing = nullptr,
                              const std::vector<Row> *forced = nullptr,
                              const std::function<void(int, const Row &)> &on_frame = {});

    // 导出每步 audio logits（teacher forcing 时用于 KL / top-k 重叠分析）
    void set_logits_dump(const std::string &path) { logits_dump_path_ = path; }

    // 导出每帧的结束判定余量（CSV: frame,logit_slot_minus_end）
    void set_gate_dump(const std::string &path) { gate_dump_path_ = path; }

    const float *hidden() const { return hidden_.data(); }
    int position() const { return pos_; }
    const std::string &info() const { return info_; }

private:
    struct Layer {
        Vector ln1_w, ln1_b, ln2_w, ln2_b;
        Matrix attn_w;
        Vector attn_b;
        Matrix proj_w;
        Vector proj_b;
        Matrix fc_in_w;
        Vector fc_in_b;
        Matrix fc_out_w;
        Vector fc_out_b;
    };

    void load_layer(Layer &layer, const std::string &prefix);
    void embed_row(const Row &row, float *out) const;
    void lookup_row(const Matrix &m, int row, float *out) const;
    void normalize(const float *x, const Vector &w, const Vector &b, float *out) const;
    void rope(float *vec, int position) const;
    void block_forward(const Layer &layer, bool local, int layer_index, int position, float *x,
                       float *qkv, float *attn, float *mlp);
    void attention_core(const float *k_cache, const float *v_cache, int n_pos, const float *q, float *out,
                        float *scores);
    void attention(const Layer &layer, bool local, int layer_index, int position, const float *qkv,
                   float *out);
    void global_step(const Row &row, float *hidden_out);
    void local_reset();
    void local_step(const float *input, float *out);
    void audio_logits(int channel, const float *hidden, float *logits);
    void text_logits(const float *hidden, float *logits);
    // 只算「是否说完」用到的两行（audio_end=7 / slot=9）：省掉 16384×768 的整表点乘，数值完全相同
    void assistant_logits(const float *hidden, float *pair);

    WeightFile wf_;
    Parallel par_;
    std::vector<Layer> global_layers_;
    Layer local_layer_;
    Vector lnf_w_, lnf_b_, local_lnf_w_, local_lnf_b_;
    Matrix wte_{};
    std::array<Matrix, kNVq> aemb_{};

    std::vector<float> g_k_, g_v_;  // [12][kMaxCtx][768]
    std::vector<float> l_k_, l_v_;  // [1][kLocalLen][768]
    // 前缀 KV 快照：[12][prefix_pos_][768]，restore_prefix() 时整体拷回 g_*
    std::vector<float> p_k_, p_v_;
    int pos_ = 0;
    int lpos_ = 0;
    int prefix_pos_ = 0;
    uint64_t weights_fp_ = 0;

    std::vector<float> x_, qkv_, attn_buf_, mlp_buf_, hidden_;
    std::vector<float> ln_buf_, res_buf_;
    // prefill 批量缓冲（按需 resize）
    std::vector<float> b_x_, b_ln_, b_qkv_, b_att_, b_mlp_, b_res_, b_scores_;
    std::vector<float> lx_, lqkv_, lattn_, lmlp_;
    std::vector<float> cos_, sin_;  // [kMaxCtx][kHidden/2]
    std::vector<float> logits_;
    std::vector<float> sample_scratch_;

    GemvScratch scratch_;
    GemvPoolScratch pool_scratch_;
    std::string info_;
    std::string logits_dump_path_;
    std::vector<float> logits_dump_;
    std::string gate_dump_path_;
    std::vector<float> gate_dump_;
};

}  // namespace moss
