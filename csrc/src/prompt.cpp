#include "moss/prompt.h"

#include <array>

namespace moss {
namespace {

// 以下三组 id 直接来自 browser_poc_manifest.json 的 prompt_templates
constexpr int kPrefixIds[] = {600, 289, 10356, 13, 10356, 10425, 1860, 4546, 12907, 10363, 13325, 11492};
constexpr int kAfterRefIds[] = {
    10356, 10425, 3965,  7738,  11492, 505,   587,   10356, 10425, 352,   500,   856,   11492, 505,   587,
    10356, 10425, 2128,  1247,  594,   11492, 505,   587,   10356, 10425, 348,   909,   561,   3648,  11492,
    505,   587,   10356, 10425, 2818,  1305,  355,   348,   909,   11492, 505,   587,   10356, 10425, 484,
    339,   10367, 783,   11492, 505,   587,   10356, 10425, 2427,  980,   11492, 505,   587};
constexpr int kAssistantIds[] = {10356, 14, 5, 4, 8165, 430};

static_assert(sizeof(kPrefixIds) / sizeof(int) == 12, "prefix template must be 12 tokens");
static_assert(sizeof(kAfterRefIds) / sizeof(int) == 58, "after-reference template must be 58 tokens");
static_assert(sizeof(kAssistantIds) / sizeof(int) == 6, "assistant template must be 6 tokens");

}  // namespace

Row make_text_row(int token) {
    Row row{};
    row.fill(kAudioPad);
    row[0] = token;
    return row;
}

int voice_prefix_rows(int voice_frames) {
    return 12 + 1 + voice_frames + 1 + 58;
}

std::vector<Row> build_voice_prefix(const int16_t *codes, int frames) {
    std::vector<Row> rows;
    rows.reserve(static_cast<size_t>(voice_prefix_rows(frames)));
    for (int id : kPrefixIds) rows.push_back(make_text_row(id));
    rows.push_back(make_text_row(kAudioStart));
    for (int f = 0; f < frames; ++f) {
        Row row{};
        row.fill(kAudioPad);
        row[0] = kUserSlot;
        for (int ch = 0; ch < kNVq; ++ch) row[ch + 1] = codes[static_cast<size_t>(f) * kNVq + ch];
        rows.push_back(row);
    }
    rows.push_back(make_text_row(kAudioEnd));
    for (int id : kAfterRefIds) rows.push_back(make_text_row(id));
    return rows;
}

std::vector<Row> build_text_suffix(const std::vector<int> &text_ids) {
    std::vector<Row> rows;
    rows.reserve(text_ids.size() + 7);
    for (int id : text_ids) rows.push_back(make_text_row(id));
    for (int id : kAssistantIds) rows.push_back(make_text_row(id));
    rows.push_back(make_text_row(kAudioStart));
    return rows;
}

}  // namespace moss
