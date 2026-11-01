#pragma once

#include <string>
#include <vector>

#include "moss/audio_layer.h"
#include "moss/common.h"
#include "moss/gemv.h"
#include "moss/parallel.h"
#include "moss/weights.h"

namespace moss {

// MOSS-Audio-Tokenizer-Nano 的解码器（整段，非流式）。
// 数据布局：激活一律 [T][C]（时间主序），权重沿用 .moss 的 [rows, cols]（rows=输出通道）。
class AudioDecoder {
public:
    // threads：本引擎独占的线程池大小（可与 LLM 不同，便于流水线分配算力）
    explicit AudioDecoder(int threads = 1);

    bool load(const std::string &path, std::string *err);

    // codes: [nq][frames]（nq 主序）；返回交织后的立体声样本（L,R,L,R...）
    std::vector<float> decode(const int32_t *codes, int frames);

    // ---- 流式 ----
    void reset_stream();
    // 解码一个块（frames 帧），返回该块的交织样本；需按顺序调用
    std::vector<float> decode_chunk(const int32_t *codes, int frames);

    double last_ms() const { return last_ms_; }
    const std::vector<double> &stage_ms() const { return stage_ms_; }
    const std::string &info() const { return info_; }

private:
    using Layer = AudioLayer;
    struct Stage {
        Matrix in_w, out_w;
        std::vector<Layer> layers;
        int patch = 1;
    };
    struct Codebook {
        Matrix cb;   // [1024, 8]
        Matrix out_w;  // [512, 8]
        Vector out_b;  // [512]
    };

    struct LayerState {
        std::vector<float> k, v;  // [n][channels]
        int n = 0;
    };
    struct StageStream {
        int offset = 0;
        int context = 0;  // 环形窗口（帧）
        std::vector<LayerState> layers;
    };

    std::vector<float> quantize_front(const int32_t *codes, int frames, int *out_t);
    void patch_forward(std::vector<float> &x, int &t, int &c, int p) const;
    void run_layer_stream(const Layer &layer, LayerState &state, int context, std::vector<float> &x, int t, int c,
                          int offset);
    void run_stage_stream(const Stage &stage, StageStream &state, std::vector<float> &x, int &t, int &c);
    void run_stage(const Stage &stage, std::vector<float> &x, int &t, int &c);
    void run_layer(const Layer &layer, std::vector<float> &x, int t, int c);

    WeightFile wf_;
    Parallel par_;
    Matrix q_out_w_;
    Vector q_out_b_;
    std::vector<Codebook> codebooks_;
    std::vector<Stage> stages_;
    GemvPoolScratch pool_;
    AudioLayerBuffers buf_;
    std::string info_;
    double last_ms_ = 0.0;
    std::vector<double> stage_ms_;
    std::vector<StageStream> stream_states_;
    int current_window_ = 0;
};

}  // namespace moss
