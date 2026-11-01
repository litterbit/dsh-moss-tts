#include "moss/parallel.h"

#include <algorithm>

namespace moss {

Parallel::Parallel(int threads) : threads_(std::max(1, threads)) {
    if (threads_ > 1) {
        barrier_ = std::make_unique<std::barrier<>>(threads_);
        for (int i = 0; i < threads_ - 1; ++i) workers_.emplace_back([this] { worker_loop(); });
    }
}

Parallel::~Parallel() {
    stop_.store(true);
    cv_.notify_all();
    for (auto &t : workers_) {
        if (t.joinable()) t.join();
    }
}

void Parallel::drain() {
    const auto &body = *body_;
    for (int i = next_.fetch_add(1); i < items_; i = next_.fetch_add(1)) body(i);
}

void Parallel::worker_loop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_.load() || pending_jobs_ > 0; });
            if (stop_.load()) return;
            --pending_jobs_;
        }
        drain();
        barrier_->arrive_and_wait();
    }
}

void Parallel::run(int items, const std::function<void(int)> &body) {
    if (items <= 0) return;
    if (threads_ <= 1 || items == 1) {
        for (int i = 0; i < items; ++i) body(i);
        return;
    }

    bool expected = false;
    if (!busy_.compare_exchange_strong(expected, true)) {
        // 重入 / 并发使用同一个池：串行跑完，不碰 barrier。
        degraded_.store(true);
        for (int i = 0; i < items; ++i) body(i);
        return;
    }

    next_.store(0);
    items_ = items;
    body_ = &body;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_jobs_ = threads_ - 1;
    }
    cv_.notify_all();

    try {
        drain();
    } catch (...) {
        barrier_->arrive_and_wait();  // 让 worker 归位，避免下次错位
        busy_.store(false);
        throw;
    }
    barrier_->arrive_and_wait();
    busy_.store(false);
}

}  // namespace moss
