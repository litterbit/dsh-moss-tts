#include "moss/engine.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <filesystem>
#include <thread>

#include "moss/prompt.h"
#include "moss/resample.h"
#include "moss/wav.h"

namespace moss {
namespace {

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 上游的分段间停顿：短句给 0.40 s，长句 0.24 s
constexpr double kPauseShortSeconds = 0.40;
constexpr double kPauseLongSeconds = 0.24;

// 参考音频上限：再长只会让 prompt 变长、编码变慢，音色本身没有额外信息。
// 20.05 s 的容差留给「剪到 20.00 s 再编码」时采样率换算的零头。
constexpr double kMaxCloneSeconds = 20.0;
constexpr double kMaxCloneTolerance = 0.05;

inline bool ends_with(const std::string &s, const char *suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && std::memcmp(s.data() + s.size() - n, suffix, n) == 0;
}

bool is_sentence_end(uint32_t cp) {
    switch (cp) {
        case U'。': case U'！': case U'？': case U'!': case U'?': case U'.':
        case U';': case U'；': case U'\n':
            return true;
        default:
            return false;
    }
}

bool is_clause_end(uint32_t cp) {
    switch (cp) {
        case U'，': case U',': case U'、': case U'：': case U':': case U'—': case U'…':
            return true;
        default:
            return false;
    }
}

// 解码一个 utf8 码点，返回长度
int decode_utf8(const std::string &s, size_t i, uint32_t *cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    int len = 1;
    uint32_t value = 0;
    if ((c & 0xE0) == 0xC0) {
        len = 2;
        value = c & 0x1F;
    } else if ((c & 0xF0) == 0xE0) {
        len = 3;
        value = c & 0x0F;
    } else if ((c & 0xF8) == 0xF0) {
        len = 4;
        value = c & 0x07;
    } else {
        *cp = c;
        return 1;
    }
    if (i + static_cast<size_t>(len) > s.size()) {
        *cp = c;
        return 1;
    }
    for (int k = 1; k < len; ++k) value = (value << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    *cp = value;
    return len;
}

// 有界帧队列：生产者（LLM 线程）满了就等，消费者（调用线程）边收边解码
class FrameQueue {
public:
    explicit FrameQueue(size_t capacity) : capacity_(capacity ? capacity : 1) {}
    void push(const Row &row) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return closed_ || queue_.size() < capacity_; });
        if (closed_) return;
        queue_.push_back(row);
        not_empty_.notify_one();
    }
    bool pop(Row *row) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return false;
        *row = queue_.front();
        queue_.pop_front();
        not_full_.notify_one();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<Row> queue_;
    size_t capacity_;
    bool closed_ = false;
};

}  // namespace

TtsEngine::~TtsEngine() = default;

bool TtsEngine::load(const EngineOptions &options, std::string *err) {
    options_ = options;
    llm_ = std::make_unique<LlmModel>(std::max(1, options.llm_threads));
    decoder_ = std::make_unique<AudioDecoder>(std::max(1, options.decode_threads));
    const auto t0 = Clock::now();
    if (!llm_->load(options.llm_path, err)) return false;
    if (!decoder_->load(options.decoder_path, err)) return false;
    const double load_ms = ms_since(t0);

    info_ = "llm=" + options.llm_path + " decoder=" + options.decoder_path;
    if (!options.tokenizer_path.empty()) {
        if (!load_tokenizer(err)) return false;
    }
    std::printf("{\"engine\":\"loaded\",\"load_ms\":%.1f,\"tokenizer\":%s}\n", load_ms,
                spm_ ? "true" : "false");
    return true;
}

bool TtsEngine::load_tokenizer(std::string *err) {
    if (spm_) return true;
    spm_ = std::make_unique<SentencePiece>();
    if (!spm_->load(options_.tokenizer_path, err)) {
        spm_.reset();
        return false;
    }
    return true;
}

bool TtsEngine::ensure_encoder(std::string *err) {
    if (encoder_) return true;
    if (options_.encoder_path.empty()) {
        if (err) *err = "没有配置编码器（--encoder）";
        return false;
    }
    encoder_ = std::make_unique<AudioEncoder>(std::max(1, options_.llm_threads));
    if (!encoder_->load(options_.encoder_path, err)) {
        encoder_.reset();
        return false;
    }
    return true;
}

std::string TtsEngine::prefix_cache_path() const {
    if (options_.cache_dir.empty() || voice_.name.empty()) return {};
    return options_.cache_dir + "/" + voice_.name + ".mosskv";
}

bool TtsEngine::set_voice(const std::string &name_or_path, std::string *err) {
    std::string path = name_or_path;
    if (!ends_with(path, ".mossclone")) {
        if (options_.voices_dir.empty()) {
            if (err) *err = "没有配置内置音色目录";
            return false;
        }
        path = options_.voices_dir + "/" + name_or_path + ".mossclone";
    }
    VoiceProfile profile;
    if (!profile_load(path, &profile, err)) return false;
    if (profile.frames <= 0 || profile.codes.empty()) {
        if (err) *err = "音色档案是空的: " + path;
        return false;
    }
    voice_ = std::move(profile);
    if (voice_.name.empty()) voice_.name = name_or_path;
    prefix_rows_ = build_voice_prefix(voice_.codes.data(), voice_.frames);
    prefix_ready_ = false;
    prefix_from_cache_ = false;
    return true;
}

bool TtsEngine::ensure_prefix_kv(std::string *err) {
    if (prefix_ready_) return true;
    if (prefix_rows_.empty()) {
        if (err) *err = "还没有设置音色";
        return false;
    }
    const std::string cache = prefix_cache_path();
    if (!cache.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(cache).parent_path(), ec);
    }
    if (!cache.empty()) {
        std::error_code exists_ec;
        const bool existed = std::filesystem::exists(cache, exists_ec);
        if (llm_->load_prefix(cache, prefix_rows_, err)) {
            prefix_ready_ = true;
            prefix_from_cache_ = true;
            return true;
        }
        // 缓存不可用（首次运行/换了权重/换了音色）就重算，并在 stderr 说明
        if (!existed) {
            std::fprintf(stderr, "[prefix] 首次使用该音色（无缓存），重算\n");
        } else {
            std::fprintf(stderr, "[prefix] 未使用缓存（%s），重算\n", err && !err->empty() ? err->c_str() : "?");
        }
    }
    llm_->prefill(prefix_rows_, 0);
    llm_->snapshot_prefix();
    if (!cache.empty()) {
        std::string save_err;
        // 目录可能不存在（发布包不带 cache），先建出来；建不出来就跳过缓存，不影响合成
        if (!cache.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(std::filesystem::path(cache).parent_path(), ec);
        }
        if (!llm_->save_prefix(cache, prefix_rows_, &save_err)) {
            std::fprintf(stderr, "[prefix] 缓存写不进去：%s\n", save_err.c_str());
        }
    }
    prefix_ready_ = true;
    prefix_from_cache_ = false;
    return true;
}

bool TtsEngine::clone_audio(const std::string &wav_path, const std::string &out_path, std::string *err) {
    if (!ensure_encoder(err)) return false;

    std::vector<float> pcm;
    int sample_rate = 0;
    int channels = 0;
    if (!wav_read(wav_path, &pcm, &sample_rate, &channels, err)) return false;
    const int in_frames = static_cast<int>(pcm.size() / static_cast<size_t>(channels));
    const double seconds = static_cast<double>(in_frames) / static_cast<double>(sample_rate);
    if (seconds > kMaxCloneSeconds + kMaxCloneTolerance) {
        if (err) {
            char buf[192];
            std::snprintf(buf, sizeof(buf), "参考音频最长 %.0f 秒（这段 %.1f 秒），请先裁剪", kMaxCloneSeconds,
                          seconds);
            *err = buf;
        }
        return false;
    }
    if (channels == 1) {
        std::vector<float> stereo(static_cast<size_t>(in_frames) * 2);
        for (int i = 0; i < in_frames; ++i) {
            stereo[static_cast<size_t>(i) * 2] = pcm[static_cast<size_t>(i)];
            stereo[static_cast<size_t>(i) * 2 + 1] = pcm[static_cast<size_t>(i)];
        }
        pcm.swap(stereo);
        channels = 2;
    } else if (channels > 2) {
        std::vector<float> stereo(static_cast<size_t>(in_frames) * 2);
        for (int i = 0; i < in_frames; ++i) {
            stereo[static_cast<size_t>(i) * 2] = pcm[static_cast<size_t>(i) * channels];
            stereo[static_cast<size_t>(i) * 2 + 1] = pcm[static_cast<size_t>(i) * channels + 1];
        }
        pcm.swap(stereo);
        channels = 2;
    }
    if (sample_rate != 48000) {
        pcm = resample_interleaved(pcm.data(), in_frames, channels, sample_rate, 48000);
        sample_rate = 48000;
    }

    std::vector<int32_t> codes;
    const int frames = encoder_->encode(pcm.data(), static_cast<int>(pcm.size() / 2), &codes);
    if (frames <= 0) {
        if (err) *err = "参考音频太短";
        return false;
    }
    VoiceProfile profile;
    profile.name = "custom";
    profile.frames = frames;
    profile.sample_rate = sample_rate;
    profile.codes.resize(static_cast<size_t>(frames) * 16);
    for (size_t i = 0; i < profile.codes.size(); ++i) profile.codes[i] = static_cast<int16_t>(codes[i]);
    return profile_save(out_path, profile, err);
}

std::vector<int32_t> TtsEngine::text_to_ids(const std::string &text) const {
    if (!spm_) return {};
    return spm_->encode(text);
}

std::vector<std::string> TtsEngine::split_text(const std::string &text, int max_tokens) const {
    std::vector<std::string> chunks;
    if (text.empty()) return chunks;
    if (!spm_) {
        chunks.push_back(text);
        return chunks;
    }
    const int budget = std::max(1, max_tokens);

    // 1) 先按句末标点切成句子（保留标点）
    std::vector<std::string> sentences;
    std::string current;
    for (size_t i = 0; i < text.size();) {
        uint32_t cp = 0;
        const int len = decode_utf8(text, i, &cp);
        current.append(text, i, static_cast<size_t>(len));
        i += static_cast<size_t>(len);
        if (is_sentence_end(cp)) {
            sentences.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) sentences.push_back(current);

    // 2) 句子太长就按从句标点再切，还不行就二分
    std::vector<std::string> pieces;
    for (const std::string &sentence : sentences) {
        if (static_cast<int>(spm_->encode(sentence).size()) <= budget) {
            pieces.push_back(sentence);
            continue;
        }
        std::string clause;
        std::vector<std::string> clauses;
        for (size_t i = 0; i < sentence.size();) {
            uint32_t cp = 0;
            const int len = decode_utf8(sentence, i, &cp);
            clause.append(sentence, i, static_cast<size_t>(len));
            i += static_cast<size_t>(len);
            if (is_clause_end(cp)) {
                clauses.push_back(clause);
                clause.clear();
            }
        }
        if (!clause.empty()) clauses.push_back(clause);
        for (const std::string &cl : clauses) {
            if (static_cast<int>(spm_->encode(cl).size()) <= budget) {
                pieces.push_back(cl);
                continue;
            }
            // 二分：按 token 预算找最长的前缀
            std::string rest = cl;
            while (!rest.empty()) {
                if (static_cast<int>(spm_->encode(rest).size()) <= budget) {
                    pieces.push_back(rest);
                    break;
                }
                size_t lo = 1, hi = rest.size();
                size_t best = 1;
                while (lo <= hi) {
                    const size_t mid = (lo + hi) / 2;
                    if (static_cast<int>(spm_->encode(rest.substr(0, mid)).size()) <= budget) {
                        best = mid;
                        lo = mid + 1;
                    } else {
                        if (mid == 0) break;
                        hi = mid - 1;
                    }
                }
                pieces.push_back(rest.substr(0, best));
                rest = rest.substr(best);
            }
        }
    }

    // 3) 把句子拼成不超过预算的块
    std::string acc;
    for (const std::string &piece : pieces) {
        if (acc.empty()) {
            acc = piece;
            continue;
        }
        const std::string joined = acc + piece;
        if (static_cast<int>(spm_->encode(joined).size()) <= budget) {
            acc = joined;
        } else {
            chunks.push_back(acc);
            acc = piece;
        }
    }
    if (!acc.empty()) chunks.push_back(acc);
    return chunks;
}

bool TtsEngine::synthesize(const std::string &text, const SynthOptions &options, SynthStats *stats,
                           std::string *err) {
    SynthStats local;
    const auto t_all = Clock::now();
    if (!ensure_prefix_kv(err)) return false;
    local.prefix_cache_hit = prefix_from_cache_;
    if (!spm_) {
        if (err) *err = "没有加载分词器（--tokenizer）";
        return false;
    }

    const std::vector<std::string> chunks = split_text(text, options.max_text_tokens);
    if (chunks.empty()) {
        if (err) *err = "文本为空";
        return false;
    }

    SampleParams params;
    params.greedy = options.greedy;
    params.end_greedy = true;
    params.min_frames = std::max(0, options.min_frames);
    params.greedy_frames = std::max(0, options.greedy_frames);
    params.temperature = options.temperature;
    params.top_k = options.top_k;
    params.top_p = options.top_p;
    params.repetition_penalty = options.repetition_penalty;

    const int chunk_frames = std::max(1, options.chunk_frames);
    const int prefix_pos = static_cast<int>(prefix_rows_.size());
    const int sample_rate = 48000;

    // 取消：LLM 每帧检查一次；解码在块边界检查（1 块 = chunk_frames 帧，默认 8 帧）。
    // prefill 不可中断（≈170 ms），这是取消延迟的上界来源之一。
    llm_->set_cancel_flag(options.cancel);
    // 解码器是流式的：每次合成都必须从干净的窗口开始，否则上一句的 KV 会污染下一句。
    // 常驻进程里这是正确性（可复现性）的前提，第一次调用时等于空操作。
    decoder_->reset_stream();
    auto stop_requested = [&]() {
        return options.cancel != nullptr && options.cancel->load(std::memory_order_relaxed);
    };

    auto emit_silence = [&](double seconds) {
        if (!options.on_audio || seconds <= 0.0) return;
        const int frames = static_cast<int>(seconds * sample_rate);
        std::vector<float> zeros(static_cast<size_t>(frames) * 2, 0.0f);
        options.on_audio(zeros.data(), frames);
        local.audio_seconds += seconds;
    };

    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        if (stop_requested()) {
            local.cancelled = true;
            break;
        }
        const std::vector<int32_t> ids = spm_->encode(chunks[ci]);
        if (ids.empty()) continue;
        local.text_tokens += static_cast<int>(ids.size());
        ++local.chunks;

        // 正文行：正文 token + assistant 前缀 + audio_start
        const std::vector<Row> suffix_rows = build_text_suffix(ids);
        const auto t_prefix = Clock::now();
        llm_->restore_prefix();
        local.prefix_ms += ms_since(t_prefix);
        const auto t_pre = Clock::now();
        llm_->prefill(suffix_rows, prefix_pos);
        local.prefill_ms += ms_since(t_pre);

        FrameQueue queue(static_cast<size_t>(chunk_frames) * 4);
        std::exception_ptr producer_error;
        std::thread producer([&] {
            try {
                llm_->generate(options.max_frames, params, options.seed, nullptr, nullptr,
                               [&](int, const Row &row) { queue.push(row); });
            } catch (...) {
                producer_error = std::current_exception();
            }
            queue.close();
        });

        std::vector<Row> pending;
        std::vector<int32_t> codes;
        auto decode_pending = [&](bool force) {
            const int n = static_cast<int>(pending.size());
            if (n == 0) return;
            // 首块更小：尽快吐出第一批音频，之后的块用正常大小。
            // 取 4 帧（320 ms 音频）而不是 2 帧：2 帧只有 160 ms 缓冲，
            // 弱机上生成下一块（8 帧）要 150~250 ms，会欠载卡顿。
            const size_t want = (local.frames == 0) ? std::min<size_t>(4, static_cast<size_t>(chunk_frames))
                                                    : static_cast<size_t>(chunk_frames);
            if (!force && static_cast<size_t>(n) < want) return;
            codes.resize(static_cast<size_t>(n) * kNVq);
            for (int i = 0; i < kNVq; ++i)
                for (int j = 0; j < n; ++j) {
                    codes[static_cast<size_t>(i) * n + j] = pending[static_cast<size_t>(j)][i + 1];
                }
            const auto t0 = Clock::now();
            const std::vector<float> pcm = decoder_->decode_chunk(codes.data(), n);
            local.decode_ms += ms_since(t0);
            local.frames += n;
            local.audio_seconds += static_cast<double>(pcm.size()) / 2.0 / sample_rate;
            if (local.first_audio_ms < 0.0) local.first_audio_ms = ms_since(t_all);
            if (options.on_audio) options.on_audio(pcm.data(), static_cast<int>(pcm.size() / 2));
            pending.clear();
        };

        Row row;
        bool cancelled = false;
        while (queue.pop(&row)) {
            if (stop_requested()) {
                // 丢掉后面的帧，但仍然排空队列，否则 producer 会卡在 push 上
                cancelled = true;
                local.cancelled = true;
                continue;
            }
            pending.push_back(row);
            decode_pending(false);
        }
        producer.join();
        if (producer_error) {
            if (err) *err = "LLM 线程异常";
            return false;
        }
        // 竞态：取消标志可能是 producer 先看到、在 consumer 退出循环之后才置位的。
        // 这里再确认一次，否则一边断流一边记成 completed（取消接口就"看起来没生效"）。
        if (!cancelled && stop_requested()) {
            cancelled = true;
            local.cancelled = true;
        }
        if (cancelled) {
            pending.clear();
            break;
        }
        decode_pending(true);
        if (local.first_chunk_frames == 0) local.first_chunk_frames = local.frames;

        // 段间停顿
        if (ci + 1 < chunks.size() && !stop_requested()) {
            const int words = static_cast<int>(std::count_if(
                chunks[ci].begin(), chunks[ci].end(), [](char c) { return c == ' '; })) + 1;
            emit_silence(words <= 4 ? kPauseShortSeconds : kPauseLongSeconds);
        }
    }

    local.total_ms = ms_since(t_all);
    llm_->set_cancel_flag(nullptr);
    if (stats) *stats = local;
    return true;
}

}  // namespace moss
