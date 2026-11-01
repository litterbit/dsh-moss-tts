#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace moss {

// SentencePiece（BPE + NFKC charsmap）的只读实现，零依赖。
// 模型由 tools/spm_dump.py 转成 .mossvocab；这里只负责 normalize + BPE + id 映射。
class SentencePiece {
public:
    bool load(const std::string &path, std::string *err);

    // NFKC + 空白规整 + ▁ 转义（与 sentencepiece 的 Normalizer 等价）
    std::string normalize(const std::string &text) const;

    // 文本 -> token id（内部先 normalize 再 BPE）
    std::vector<int32_t> encode(const std::string &text) const;

    // id -> 文本（调试用；▁ 还原成空格）
    std::string decode(const std::vector<int32_t> &ids) const;

    std::string piece(int32_t id) const;
    int unk_id() const { return unk_; }
    int vocab_size() const { return static_cast<int>(pieces_.size()); }

private:
    struct Piece {
        std::string bytes;
        float score = 0.0f;
        uint8_t type = 0;
    };

    bool prefix_match(const char *key, int len, int *matched, int *value) const;

    std::vector<Piece> pieces_;
    std::unordered_map<std::string, int32_t> by_bytes_;
    std::array<int32_t, 256> byte_pieces_{};
    std::vector<uint32_t> trie_;
    std::string blob_;
    int unk_ = 0, bos_ = 1, eos_ = 2, pad_ = 3;
    uint32_t flags_ = 0;
};

}  // namespace moss
