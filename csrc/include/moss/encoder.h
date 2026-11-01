#pragma once

#include <string>
#include <vector>

#include "moss/audio_layer.h"
#include "moss/common.h"
#include "moss/gemv.h"
#include "moss/parallel.h"
#include "moss/weights.h"

namespace moss {

// MOSS-Audio-Tokenizer 的编码器：48 kHz 立体声 -> 16 路离散码，80 ms 一帧。
// 与解码器互逆（同一套 AudioLayer），额外多一个残差 LFQ 的正向量化。
class AudioEncoder {
public:
    explicit AudioEncoder(int threads = 1);

    bool load(const std::string &path, std::string *err);

    // pcm 是**交织**立体声（L,R,L,R...），per_channel 是每声道样本数。
    // codes 输出 frame-major：[frames][16]。返回帧数（= per_channel / 3840，与上游一致）。
    int encode(const float *pcm, int per_channel, std::vector<int32_t> *codes);

    double last_ms() const { return last_ms_; }
    const std::vector<double> &stage_ms() const { return stage_ms_; }
    const std::string &info() const { return info_; }

private:
    struct Stage {
        Matrix in_w, out_w;
        std::vector<AudioLayer> layers;
        int patch = 1;
        int window = 0;  // 因果窗口（token 数），0 = 不限
    };
    struct Lfq {
        Matrix in_w;   // [8, 512]
        Vector in_b;
        Matrix out_w;  // [512, 8]
        Vector out_b;
        std::vector<float> cb_raw;   // [1024][8] 原始码本（重建残差用）
        std::vector<float> cb_norm;  // [1024][8] L2 归一化（最近邻搜索用）
    };

    // 下采样 patch：out[ti][ci*p + k] = in[ti*p + k][ci]
    void patch_down(std::vector<float> &x, int &t, int &c, int p) const;
    void run_stage(const Stage &stage, std::vector<float> &x, int &t, int &c);
    void run_quantizer(std::vector<float> &x, int t, std::vector<int32_t> *codes);

    WeightFile wf_;
    Parallel par_;
    GemvPoolScratch pool_;
    AudioLayerBuffers buf_;
    Matrix q_in_w_;
    Vector q_in_b_;
    std::vector<Lfq> lfqs_;
    std::vector<Stage> stages_;
    int kDModel_ = 256;
    std::string info_;
    double last_ms_ = 0.0;
    std::vector<double> stage_ms_;
};

}  // namespace moss
