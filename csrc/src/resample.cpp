#include "moss/resample.h"

#include <algorithm>
#include <cmath>

namespace moss {
namespace {

constexpr int kHalfTaps = 16;  // 每侧 16 个抽头，共 32
constexpr double kBeta = 8.6;
constexpr double kPi = 3.14159265358979323846;

double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double y = x * x * 0.25;
    for (int k = 1; k < 32; ++k) {
        term *= y / (static_cast<double>(k) * k);
        sum += term;
        if (term < 1e-12 * sum) break;
    }
    return sum;
}

double kaiser(double t) {  // t in [-1, 1]
    const double r = 1.0 - t * t;
    if (r <= 0.0) return 0.0;
    return bessel_i0(kBeta * std::sqrt(r)) / bessel_i0(kBeta);
}

}  // namespace

std::vector<float> resample_interleaved(const float *in, int frames, int channels, int in_rate, int out_rate) {
    if (frames <= 0 || channels <= 0 || in_rate <= 0 || out_rate <= 0) return {};
    if (in_rate == out_rate) return std::vector<float>(in, in + static_cast<size_t>(frames) * channels);

    const int64_t out_frames = static_cast<int64_t>(frames) * out_rate / in_rate;
    std::vector<float> out(static_cast<size_t>(out_frames) * channels, 0.0f);
    const double ratio = static_cast<double>(in_rate) / static_cast<double>(out_rate);
    // 截止频率（相对输入采样率的归一化值）：升采样保持 0.5，降采样按比例收缩
    const double fc = 0.5 * std::min(1.0, static_cast<double>(out_rate) / static_cast<double>(in_rate)) * 0.98;

    std::vector<double> weights(2 * kHalfTaps);
    for (int64_t n = 0; n < out_frames; ++n) {
        const double src = static_cast<double>(n) * ratio;
        const int64_t center = static_cast<int64_t>(std::floor(src));
        const double frac = src - static_cast<double>(center);

        double sum = 0.0;
        for (int k = 0; k < 2 * kHalfTaps; ++k) {
            const double d = static_cast<double>(k - kHalfTaps + 1) - frac;  // 抽头相对位置
            double sinc = 0.0;
            if (std::fabs(d) < 1e-9) {
                sinc = 2.0 * fc;
            } else {
                sinc = std::sin(2.0 * kPi * fc * d) / (kPi * d);
            }
            const double w = kaiser(d / kHalfTaps);
            const double coeff = sinc * w;
            weights[k] = coeff;
            sum += coeff;
        }
        if (std::fabs(sum) > 1e-12) {
            for (int k = 0; k < 2 * kHalfTaps; ++k) weights[k] /= sum;
        }
        for (int ch = 0; ch < channels; ++ch) {
            double acc = 0.0;
            for (int k = 0; k < 2 * kHalfTaps; ++k) {
                const int64_t idx = center + (k - kHalfTaps + 1);
                if (idx < 0 || idx >= frames) continue;
                acc += weights[k] * in[idx * channels + ch];
            }
            out[static_cast<size_t>(n) * channels + ch] = static_cast<float>(acc);
        }
    }
    return out;
}

}  // namespace moss
