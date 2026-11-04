#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "moss/decoder.h"
#include "moss/encoder.h"
#include "moss/llm.h"
#include "moss/profile.h"
#include "moss/spm.h"

namespace moss {

struct EngineOptions {
    std::string llm_path;
    std::string decoder_path;
    std::string encoder_path;    // 可空（不克隆就不加载）
    std::string tokenizer_path;  // .mossvocab
    std::string voices_dir;      // 内置音色的 .mossclone 目录
    std::string cache_dir;       // 前缀 KV 缓存目录（空 = 不落盘）
    int llm_threads = 4;
    int decode_threads = 6;
};

struct SynthOptions {
    std::string voice = "Junhao";
    std::string profile_path;  // 非空则用它，忽略 voice
    uint64_t seed = 0;
    bool greedy = false;
    int max_frames = 375;   // 每段帧上限（0 = 只受上下文限制）
    int chunk_frames = 8;   // 解码块大小（首块固定 2 帧，见 engine.cpp）
    int min_frames = 0;     // 至少这么多帧才允许结束
    int greedy_frames = 0;  // 前 N 帧强制贪心（默认 0：全部走采样，起音问题根因是 prompt 模板）
    // 采样参数（只影响 16 个音频 token；结束判定始终贪心）
    float temperature = 0.8f;
    int top_k = 25;
    float top_p = 0.95f;
    float repetition_penalty = 1.2f;
    int max_text_tokens = 75;
    // 取消标志：非空时每帧检查，置位就提前收尾。检查粒度 = 1 个生成帧（或 1 个 prefill）。
    const std::atomic<bool> *cancel = nullptr;
    // 交织 float32 音频回调，在调用线程上同步调用
    std::function<void(const float *, int)> on_audio;
};

struct SynthStats {
    int text_tokens = 0;
    int chunks = 0;
    int frames = 0;
    int first_chunk_frames = 0;
    double prefix_ms = 0.0;
    double prefill_ms = 0.0;
    double generate_ms = 0.0;
    double decode_ms = 0.0;
    double first_audio_ms = -1.0;
    double total_ms = 0.0;
    double audio_seconds = 0.0;
    bool prefix_cache_hit = false;
    bool cancelled = false;
};

// 端到端引擎：文本 -> token -> prompt -> LLM 出码 -> 解码器流式出音。
// 音色（克隆）是 prompt 前缀；前缀 KV 会缓存，正文每段只算自己那几十行。
class TtsEngine {
public:
    ~TtsEngine();

    bool load(const EngineOptions &options, std::string *err);

    bool set_voice(const std::string &name_or_path, std::string *err);
    const VoiceProfile &voice() const { return voice_; }

    // 参考音频 -> .mossclone（会懒加载编码器）
    bool clone_audio(const std::string &wav_path, const std::string &out_path, std::string *err);

    std::vector<int32_t> text_to_ids(const std::string &text) const;
    std::vector<std::string> split_text(const std::string &text, int max_tokens) const;

    bool synthesize(const std::string &text, const SynthOptions &options, SynthStats *stats, std::string *err);

    const std::string &info() const { return info_; }

private:
    bool load_tokenizer(std::string *err);
    bool ensure_encoder(std::string *err);
    bool ensure_prefix_kv(std::string *err);
    std::string prefix_cache_path() const;

    EngineOptions options_;
    std::unique_ptr<LlmModel> llm_;
    std::unique_ptr<AudioDecoder> decoder_;
    std::unique_ptr<AudioEncoder> encoder_;
    std::unique_ptr<SentencePiece> spm_;
    VoiceProfile voice_;
    std::vector<Row> prefix_rows_;
    bool prefix_ready_ = false;
    bool prefix_from_cache_ = false;
    std::string info_;
};

}  // namespace moss
