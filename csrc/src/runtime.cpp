#include "moss/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "moss/cpu_features.h"
#include "moss/json.h"
#include "moss/version.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace moss {
namespace {

constexpr int kKeepJobs = 32;
constexpr int kJobTtlSeconds = 600;
constexpr size_t kMaxTextChars = 4000;
const char *kDefaultVoice = "Junhao";

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline int16_t to_pcm16(float v) {
    const float clamped = std::max(-1.0f, std::min(1.0f, v));
    return static_cast<int16_t>(std::lrintf(clamped * 32767.0f));
}

std::string read_text_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// 按物理核给两套线程数：解码器是热点，可以比 LLM 多给
void auto_threads(int *llm, int *decode) {
    int cores = cpu_features().physical_cores;
    if (cores <= 0) cores = 1;
    int l = 4, d = 6;
    if (cores <= 2) {
        l = 2;
        d = 2;
    } else if (cores <= 4) {
        l = 3;
        d = 4;
    } else if (cores <= 6) {
        l = 4;
        d = 4;
    }
    const int logical = std::max(1, cpu_features().logical_cores);
    l = std::min(l, logical);
    d = std::min(d, logical);
    *llm = l;
    *decode = d;
}

}  // namespace

struct Runtime::Job {
    std::string id;
    JobRequest request;
    PcmSink sink;
    std::atomic<bool> cancel{false};
    JobInfo info;
    Clock::time_point created = Clock::now();
    Clock::time_point finished;
};

const char *job_state_name(JobState state) {
    switch (state) {
        case JobState::Queued: return "queued";
        case JobState::Running: return "running";
        case JobState::Completed: return "completed";
        case JobState::Cancelled: return "cancelled";
        case JobState::Failed: return "failed";
    }
    return "unknown";
}

void trim_tail_silence(std::vector<int16_t> *pcm, int sample_rate, int channels) {
    if (pcm == nullptr || sample_rate <= 0 || channels <= 0) return;
    const size_t per = static_cast<size_t>(0.08 * sample_rate) * static_cast<size_t>(channels);
    if (per == 0 || pcm->size() < per * 3) return;

    const size_t frames = pcm->size() / per;
    std::vector<double> db;
    db.reserve(frames);
    for (size_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        const size_t base = i * per;
        for (size_t k = 0; k < per; ++k) {
            const double v = static_cast<double>((*pcm)[base + k]) / 32768.0;
            acc += v * v;
        }
        const double rms = std::sqrt(acc / static_cast<double>(per));
        db.push_back(20.0 * std::log10(std::max(rms, 1e-9)));
    }

    const double peak = *std::max_element(db.begin(), db.end());
    const double threshold = std::max(-50.0, peak - 28.0);
    long last = -1;
    for (size_t i = 0; i < db.size(); ++i) {
        if (db[i] >= threshold) last = static_cast<long>(i);
    }
    if (last < 0) {
        pcm->clear();
        return;
    }
    const size_t keep = std::min(pcm->size(),
                                 (static_cast<size_t>(last) + 1) * per +
                                     static_cast<size_t>(0.15 * sample_rate) * static_cast<size_t>(channels));
    if (keep >= pcm->size()) return;
    if (pcm->size() - keep < static_cast<size_t>(0.1 * sample_rate) * static_cast<size_t>(channels)) return;
    pcm->resize(keep);
}

Runtime::~Runtime() { shutdown(); }

bool Runtime::load(const RuntimeOptions &options, std::string *err) {
    options_ = options;
    if (options_.max_queue < 1) options_.max_queue = 1;
    if (options_.max_queue > 16) options_.max_queue = 16;

    int llm_threads = options_.llm_threads;
    int decode_threads = options_.decode_threads;
    if (llm_threads <= 0 || decode_threads <= 0) {
        int auto_llm = 0, auto_decode = 0;
        auto_threads(&auto_llm, &auto_decode);
        if (llm_threads <= 0) llm_threads = auto_llm;
        if (decode_threads <= 0) decode_threads = auto_decode;
    }
    options_.llm_threads = llm_threads;
    options_.decode_threads = decode_threads;

    EngineOptions eo;
    eo.llm_path = options_.llm_path;
    eo.decoder_path = options_.decoder_path;
    eo.encoder_path = options_.encoder_path;
    eo.tokenizer_path = options_.tokenizer_path;
    eo.voices_dir = options_.voices_dir;
    eo.cache_dir = options_.cache_dir;
    eo.llm_threads = llm_threads;
    eo.decode_threads = decode_threads;
    if (!engine_.load(eo, err)) return false;

    if (!options_.manifest_path.empty()) {
        const std::string text = read_text_file(options_.manifest_path);
        JsonValue root;
        std::string parse_err;
        if (!text.empty() && json_parse(text, &root, &parse_err)) {
            if (const JsonValue *v = root.get("model_id")) model_id_ = v->as_string();
            if (const JsonValue *v = root.get("model_revision")) model_revision_ = v->as_string();
        }
    }

    std::ostringstream info;
    info << "\"ok\":true,\"version\":\"" << MOSS_VERSION << "\",\"api\":\"" << MOSS_API_VERSION << "\"";
    info << ",\"model\":{\"id\":\"" << json_escape(model_id_) << "\",\"revision\":\""
         << json_escape(model_revision_) << "\"}";
    info << ",\"engine\":{\"llm_threads\":" << llm_threads << ",\"decode_threads\":" << decode_threads
         << ",\"backend\":\"" << cpu_features().gemv_backend() << "\",\"physical_cores\":"
         << cpu_features().physical_cores << "}";
    info_ = info.str();

    worker_ = std::thread([this] { worker_loop(); });
    return true;
}

void Runtime::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) return;
        stop_ = true;
        for (auto &job : queue_) {
            job->info.state = JobState::Cancelled;
            job->info.error = "shutting down";
            job->cancel.store(true);
        }
        queue_.clear();
    }
    job_added_.notify_all();
    if (worker_.joinable()) worker_.join();
}

std::string Runtime::info_json() const {
    std::ostringstream out;
    out << "{" << info_;
    out << ",\"queue\":{\"depth\":" << queue_depth() << ",\"max\":" << options_.max_queue << "}";
    out << "}";
    return out.str();
}

std::vector<VoiceInfo> Runtime::voices() const {
    std::vector<VoiceInfo> out;
    auto scan = [&](const std::string &dir, bool user) {
        if (dir.empty()) return;
        std::error_code ec;
        fs::directory_iterator it(dir, ec);
        if (ec) return;
        std::vector<VoiceInfo> found;
        for (const auto &entry : it) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".mossclone") continue;
            VoiceInfo info;
            info.id = entry.path().stem().string();
            info.user = user;
            std::ifstream in(entry.path(), std::ios::binary);
            char head[16] = {};
            if (in.read(head, sizeof(head))) {
                if (std::memcmp(head, "MSCL", 4) == 0) {
                    int32_t frames = 0;
                    std::memcpy(&frames, head + 12, 4);
                    info.frames = frames;
                    info.seconds = static_cast<double>(frames) * 0.08;
                }
            }
            found.push_back(std::move(info));
        }
        std::sort(found.begin(), found.end(), [](const VoiceInfo &a, const VoiceInfo &b) { return a.id < b.id; });
        out.insert(out.end(), found.begin(), found.end());
    };
    scan(options_.voices_dir, false);
    scan(options_.user_voices_dir, true);
    return out;
}

bool Runtime::clone_voice(const std::string &name, const std::vector<uint8_t> &wav_bytes, std::string *err) {
    if (name.empty() || name.size() > 32) {
        if (err) *err = "音色名长度不合法";
        return false;
    }
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                        c == '-';
        if (!ok) {
            if (err) *err = "音色名只能包含字母、数字、下划线和短横线";
            return false;
        }
    }
    if (options_.user_voices_dir.empty()) {
        if (err) *err = "没有配置用户音色目录";
        return false;
    }
    std::error_code ec;
    fs::create_directories(options_.user_voices_dir, ec);

    const fs::path tmp = fs::path(options_.cache_dir.empty() ? fs::temp_directory_path() : fs::path(options_.cache_dir)) /
                         ("clone_" + name + ".tmp.wav");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            if (err) *err = "写临时文件失败";
            return false;
        }
        out.write(reinterpret_cast<const char *>(wav_bytes.data()), static_cast<std::streamsize>(wav_bytes.size()));
    }
    const fs::path target = fs::path(options_.user_voices_dir) / (name + ".mossclone");
    const bool ok = engine_.clone_audio(tmp.string(), target.string(), err);
    std::error_code rm_ec;
    fs::remove(tmp, rm_ec);
    return ok;
}

std::string Runtime::make_job_id() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++counter_;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%08llx%04llx", static_cast<unsigned long long>(counter_),
                  static_cast<unsigned long long>(Clock::now().time_since_epoch().count() & 0xFFFF));
    return buf;
}

bool Runtime::submit(const JobRequest &request, PcmSink sink, std::string *job_id, std::string *err) {
    if (request.text.empty()) {
        if (err) *err = "文本为空";
        return false;
    }
    if (request.text.size() > kMaxTextChars) {
        if (err) *err = "文本过长（上限 4000 字符）";
        return false;
    }
    auto job = std::make_shared<Job>();
    job->id = request.id.empty() ? make_job_id() : request.id;
    job->request = request;
    if (job->request.voice.empty()) job->request.voice = kDefaultVoice;
    job->sink = std::move(sink);
    job->info.id = job->id;
    job->info.state = JobState::Queued;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) {
            if (err) *err = "服务正在退出";
            return false;
        }
        if (static_cast<int>(queue_.size()) >= options_.max_queue) {
            if (err) *err = "busy";
            return false;
        }
        queue_.push_back(job);
        jobs_.push_back(job);
        prune_jobs_locked();
    }
    job_added_.notify_one();
    if (job_id) *job_id = job->id;
    return true;
}

void Runtime::worker_loop() {
    while (true) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            job_added_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            job = queue_.front();
            queue_.pop_front();
        }
        if (job->cancel.load()) {
            finish_job(job, JobState::Cancelled, "cancelled");
            continue;
        }
        run_job(job);
    }
}

void Runtime::run_job(const std::shared_ptr<Job> &job) {
    const JobRequest &req = job->request;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job->info.state = JobState::Running;
    }
    job_done_.notify_all();

    const auto started = Clock::now();
    std::string error;
    SynthOptions options;
    options.voice = req.voice;
    options.profile_path = req.profile;
    options.seed = req.seed;
    options.greedy = req.greedy;
    options.max_frames = req.max_frames > 0 ? req.max_frames : 375;
    options.chunk_frames = req.chunk_frames > 0 ? req.chunk_frames : 8;
    options.min_frames = std::max(0, req.min_frames);
    if (req.temperature > 0.0f) options.temperature = req.temperature;
    if (req.top_k > 0) options.top_k = req.top_k;
    if (req.top_p > 0.0f) options.top_p = req.top_p;
    if (req.repetition_penalty > 0.0f) options.repetition_penalty = req.repetition_penalty;
    options.cancel = &job->cancel;

    std::vector<int16_t> pcm;
    int sink_error = 0;
    options.on_audio = [&](const float *data, int frames) {
        if (frames <= 0) return;
        pcm.resize(static_cast<size_t>(frames) * 2);
        for (int i = 0; i < frames * 2; ++i) pcm[static_cast<size_t>(i)] = to_pcm16(data[i]);
        if (job->sink && !job->sink(pcm.data(), frames)) {
            // 接收端出错（例如浏览器断开）：让引擎在下一帧边界停下来
            sink_error = 1;
            job->cancel.store(true);
        }
    };

    SynthStats stats;
    bool ok = false;
    try {
        if (!req.profile.empty()) {
            ok = engine_.set_voice(req.profile, &error);
        } else {
            ok = engine_.set_voice(req.voice, &error);
        }
        if (ok) ok = engine_.synthesize(req.text, options, &stats, &error);
    } catch (const std::exception &e) {
        ok = false;
        error = e.what();
    } catch (...) {
        ok = false;
        error = "未知异常";
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        job->info.frames = stats.frames;
        job->info.text_tokens = stats.text_tokens;
        job->info.audio_ms = stats.audio_seconds * 1000.0;
        job->info.compute_ms = ms_since(started);
        job->info.first_pcm_ms = stats.first_audio_ms;
        job->info.cache_hit = stats.prefix_cache_hit;
        job->info.rtf = job->info.audio_ms > 0.0 ? job->info.compute_ms / job->info.audio_ms : 0.0;
    }

    if (!ok) {
        finish_job(job, JobState::Failed, error.empty() ? "推理失败" : error);
    } else if (stats.cancelled || sink_error != 0) {
        finish_job(job, JobState::Cancelled, "cancelled");
    } else {
        finish_job(job, JobState::Completed, "");
    }
}

void Runtime::finish_job(const std::shared_ptr<Job> &job, JobState state, const std::string &error) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job->info.state = state;
        job->info.error = error;
        job->finished = Clock::now();
    }
    job_done_.notify_all();
}

void Runtime::prune_jobs_locked() {
    const auto now = Clock::now();
    jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(),
                               [&](const std::shared_ptr<Job> &job) {
                                   const bool terminal = job->info.state == JobState::Completed ||
                                                         job->info.state == JobState::Cancelled ||
                                                         job->info.state == JobState::Failed;
                                   if (!terminal) return false;
                                   const double age =
                                       std::chrono::duration<double>(now - job->finished).count();
                                   return age > kJobTtlSeconds;
                               }),
                jobs_.end());
    while (static_cast<int>(jobs_.size()) > kKeepJobs) {
        const bool terminal = jobs_.front()->info.state == JobState::Completed ||
                              jobs_.front()->info.state == JobState::Cancelled ||
                              jobs_.front()->info.state == JobState::Failed;
        if (!terminal) break;
        jobs_.erase(jobs_.begin());
    }
}

bool Runtime::wait(const std::string &job_id, int timeout_ms, JobInfo *out) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto find = [&]() -> std::shared_ptr<Job> {
        for (auto &job : jobs_) {
            if (job->id == job_id) return job;
        }
        return nullptr;
    };
    auto job = find();
    if (!job) return false;
    const auto deadline = Clock::now() + std::chrono::milliseconds(std::max(0, timeout_ms));
    job_done_.wait_until(lock, deadline, [&] {
        const auto state = job->info.state;
        return state == JobState::Completed || state == JobState::Cancelled || state == JobState::Failed;
    });
    if (out) *out = job->info;
    const auto state = job->info.state;
    return state == JobState::Completed || state == JobState::Cancelled || state == JobState::Failed;
}

bool Runtime::job(const std::string &job_id, JobInfo *out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &job : jobs_) {
        if (job->id == job_id) {
            if (out) *out = job->info;
            return true;
        }
    }
    return false;
}

bool Runtime::cancel(const std::string &job_id) {
    std::shared_ptr<Job> target;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &job : jobs_) {
            if (job->id == job_id) {
                target = job;
                break;
            }
        }
    }
    if (!target) return false;
    target->cancel.store(true);
    return true;
}

int Runtime::queue_depth() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(queue_.size());
}

}  // namespace moss
