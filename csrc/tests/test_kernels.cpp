// 内核自检：量化往返、标量 vs AVX2 对拍、与 fp32 参考的误差。
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "moss/cpu_features.h"
#include "moss/fp16.h"
#include "moss/gemv.h"
#include "moss/parallel.h"
#include "moss/quant.h"

using namespace moss;

namespace {

std::vector<BlockQ8_0> quantize_matrix_q8(const std::vector<float> &w, int rows, int cols) {
    const int nb = cols / 32;
    std::vector<BlockQ8_0> out(static_cast<size_t>(rows) * nb);
    for (int r = 0; r < rows; ++r) {
        quantize_row_q8_0(w.data() + static_cast<size_t>(r) * cols, out.data() + static_cast<size_t>(r) * nb, cols);
    }
    return out;
}

std::vector<BlockQ4_0> quantize_matrix_q4(const std::vector<float> &w, int rows, int cols) {
    const int nb = cols / 32;
    std::vector<BlockQ4_0> out(static_cast<size_t>(rows) * nb);
    for (int r = 0; r < rows; ++r) {
        const float *row = w.data() + static_cast<size_t>(r) * cols;
        for (int b = 0; b < nb; ++b) {
            const float *src = row + b * 32;
            float amax = 0.0f, max_signed = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float v = src[i];
                if (std::fabs(v) > amax) {
                    amax = std::fabs(v);
                    max_signed = v;
                }
            }
            const float d = max_signed / -8.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            BlockQ4_0 &blk = out[static_cast<size_t>(r) * nb + b];
            blk.d = float_to_half(d);
            for (int j = 0; j < 16; ++j) {
                const int q0 = static_cast<int>(std::lrintf(src[2 * j] * id + 8.5f));
                const int q1 = static_cast<int>(std::lrintf(src[2 * j + 1] * id + 8.5f));
                const int c0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
                const int c1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
                blk.qs[j] = static_cast<uint8_t>((c1 << 4) | c0);
            }
        }
    }
    return out;
}

double rel_error(const std::vector<float> &got, const std::vector<float> &ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(got[i]) - ref[i];
        num += d * d;
        den += static_cast<double>(ref[i]) * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1.0));
}

}  // namespace

int main() {
    Parallel par(4);
    const CpuFeatures &f = cpu_features();
    std::printf("cpu: avx2=%d f16c=%d avx_vnni=%d threads=%d brand=%s\n", f.avx2, f.f16c, f.avx_vnni,
                par.threads(), f.brand.c_str());

    const int rows = 2304;
    const int cols = 768;
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.05f);

    std::vector<float> w(static_cast<size_t>(rows) * cols);
    std::vector<float> x(cols);
    for (auto &v : w) v = dist(rng);
    for (auto &v : x) v = dist(rng) * 8.0f;

    std::vector<float> y_ref(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int c = 0; c < cols; ++c) acc += static_cast<double>(w[static_cast<size_t>(r) * cols + c]) * x[c];
        y_ref[r] = static_cast<float>(acc);
    }

    GemvScratch scratch;
    scratch.ensure(cols);
    quantize_row_q8_0_scaled(x.data(), scratch.act.blocks.data(), scratch.act.scales.data(), cols);

    auto check = [&](const char *tag, const Matrix &m, float *out) {
        std::vector<float> a(rows), b(rows);
        gemv_scalar_range(m, x.data(), &scratch.act, a.data(), 0, rows, scratch);
        std::printf("%-6s scalar rel=%.6f", tag, rel_error(a, y_ref));
        if (f.avx2) {
            // AVX2 的 Q4 内核读 even/odd 拆分，必须先走 quantize_activation（只做 Q8 行量化是不够的）
            if (m.dtype == DType::Q4_0) quantize_activation(x.data(), scratch.act, cols);
            gemv_avx2_range(m, x.data(), &scratch.act, b.data(), 0, rows, scratch);
            std::printf("  avx2 rel=%.6f  scalar-vs-avx2=%.2e", rel_error(b, y_ref), rel_error(b, a));
            std::copy(b.begin(), b.end(), out);
        } else {
            std::copy(a.begin(), a.end(), out);
        }
        std::printf("\n");
    };

    std::vector<uint16_t> w16(w.size());
    for (size_t i = 0; i < w.size(); ++i) w16[i] = float_to_half(w[i]);
    Matrix m16{w16.data(), rows, cols, DType::F16};
    std::vector<float> y16(rows);
    check("f16", m16, y16.data());

    auto wq8 = quantize_matrix_q8(w, rows, cols);
    Matrix mq8{wq8.data(), rows, cols, DType::Q8_0};
    std::vector<float> yq8(rows);
    check("q8", mq8, yq8.data());

    auto wq4 = quantize_matrix_q4(w, rows, cols);
    Matrix mq4{wq4.data(), rows, cols, DType::Q4_0};
    std::vector<float> yq4(rows);
    check("q4", mq4, yq4.data());

    // 多线程一致性：gemv_par 应与单线程逐位相同
    GemvPoolScratch pool;
    pool.bind(par, cols);
    std::vector<float> y_par(rows);
    gemv_par(mq8, x.data(), y_par.data(), pool);
    std::printf("q8 par-vs-scalar rel=%.2e (degraded=%d)\n", rel_error(y_par, yq8), par.degraded());
    return 0;
}
