#pragma once

#include <atomic>
#include <barrier>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace moss {

// 极简线程池：固定 worker + 每轮一个 barrier，避免每步创建线程。
// 主线程也参与计算，因此 threads=4 意味着总并发 4。
//
// 每个引擎（LLM / 解码器 / 编码器）各持一份实例，由引擎自己持有、自己不共享：
// "池"和"谁在用"一一对应，就不需要 thread_local，也不会出现跨池等待。
class Parallel {
public:
    explicit Parallel(int threads);
    ~Parallel();

    Parallel(const Parallel &) = delete;
    Parallel &operator=(const Parallel &) = delete;

    int threads() const { return threads_; }

    // 把 [0, items) 平均分给所有线程执行；返回时全部完成。
    // 若同一个池被重入（body 里又调 run）或两个线程同时使用，则降级为串行执行：
    // 宁可慢一点，也绝不在 barrier 上挂死。
    void run(int items, const std::function<void(int)> &body);

    // 是否发生过降级（调试/自检用）
    bool degraded() const { return degraded_.load(); }

private:
    void worker_loop();
    void drain();

    int threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<int> next_{0};
    std::atomic<bool> stop_{false};
    std::atomic<bool> busy_{false};
    std::atomic<bool> degraded_{false};
    int pending_jobs_ = 0;
    int items_ = 0;
    const std::function<void(int)> *body_ = nullptr;
    std::unique_ptr<std::barrier<>> barrier_;
};

}  // namespace moss
