// 常驻运行时：模型生命周期 + 单工作线程任务队列 + 取消 + 统计。
//
// 传输层（HTTP / stdio）只负责收发，不碰模型和任务状态；这才是网页与 DSH 插件
// 共用同一套语义的地方。
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "moss/engine.h"

namespace moss {

struct RuntimeOptions {
    std::string llm_path;
    std::string decoder_path;
    std::string encoder_path;
    std::string tokenizer_path;
    std::string voices_dir;       // 内置音色目录
    std::string user_voices_dir;  // 用户克隆（与内置物理隔离）
    std::string cache_dir;        // 前缀 KV 缓存 + 临时文件
    std::string manifest_path;    // models/manifest.json（可空）
    int llm_threads = 0;          // 0 = 按物理核自动
    int decode_threads = 0;
    int max_queue = 4;
};

struct VoiceInfo {
    std::string id;
    bool user = false;
    int frames = 0;
    double seconds = 0.0;
};

struct JobRequest {
    std::string id;                // 客户端指定（可空，空则由运行时生成）
    std::string text;
    std::string voice = "Junhao";  // 内置音色 id
    std::string profile;           // 自定义档案路径（非空则优先于 voice）
    uint64_t seed = 0;
    bool greedy = false;
    int max_frames = 375;
    int chunk_frames = 8;
    int min_frames = 0;
    float temperature = 0.0f;  // 0 = 用引擎默认（0.8）
    int top_k = 0;             // 0 = 默认 25
    float top_p = 0.0f;        // 0 = 默认 0.95
    float repetition_penalty = 0.0f;  // 0 = 默认 1.2
    // 边播边算：允许领先播放位置多少秒（0 = 不限速，一次算完）。
    // 睡眠发生在音频块之间（一块 = chunk_frames 帧），CPU 会真正闲下来。
    double pace = 0.0;
};

enum class JobState { Queued, Running, Completed, Cancelled, Failed };

const char *job_state_name(JobState state);

struct JobInfo {
    std::string id;
    JobState state = JobState::Queued;
    int frames = 0;
    double audio_ms = 0.0;
    double compute_ms = 0.0;  // 纯计算时间（扣掉限速等待和阻塞的发送）
    double wall_ms = 0.0;     // 从提交到结束的墙钟时间
    double first_pcm_ms = -1.0;
    double rtf = 0.0;
    int text_tokens = 0;
    bool cache_hit = false;
    std::string error;
};

// 每个音频块回调一次（工作线程上同步调用）。返回 false 表示接收端出错，任务将被中断。
using PcmSink = std::function<bool(const int16_t *pcm, int frames)>;

// 尾部静音/自言自语裁剪（沿用原 web/server.py 的门限，默认 28 dB / -50 dBFS / 留 150 ms）
void trim_tail_silence(std::vector<int16_t> *pcm, int sample_rate, int channels);

class Runtime {
public:
    ~Runtime();

    bool load(const RuntimeOptions &options, std::string *err);
    void shutdown();

    const RuntimeOptions &options() const { return options_; }
    std::string info_json() const;
    const std::string &model_id() const { return model_id_; }
    const std::string &model_revision() const { return model_revision_; }

    std::vector<VoiceInfo> voices() const;
    bool clone_voice(const std::string &name, const std::vector<uint8_t> &wav_bytes, std::string *err);

    // 入队；队列满返回 false（err = "busy"）。返回的 job_id 可用于查询/取消。
    bool submit(const JobRequest &request, PcmSink sink, std::string *job_id, std::string *err);
    // 阻塞直到任务结束或超时（毫秒）
    bool wait(const std::string &job_id, int timeout_ms, JobInfo *out);
    bool job(const std::string &job_id, JobInfo *out) const;
    bool cancel(const std::string &job_id);

    int queue_depth() const;
    int max_queue() const { return options_.max_queue; }

private:
    struct Job;
    void worker_loop();
    void run_job(const std::shared_ptr<Job> &job);
    void finish_job(const std::shared_ptr<Job> &job, JobState state, const std::string &error);
    void prune_jobs_locked();
    std::string make_job_id();

    RuntimeOptions options_;
    TtsEngine engine_;
    std::string model_id_, model_revision_;
    std::string info_;

    mutable std::mutex mutex_;
    std::condition_variable job_added_;
    std::condition_variable job_done_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::vector<std::shared_ptr<Job>> jobs_;  // 保留最近若干条（TTL 清理）
    bool stop_ = false;
    std::thread worker_;
    uint64_t counter_ = 0;
};

}  // namespace moss
