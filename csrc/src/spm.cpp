#include "moss/spm.h"

#include <cstdio>
#include <cstring>
#include <limits>

namespace moss {
namespace {

constexpr char kMagic[8] = {'M', 'O', 'S', 'V', 'O', 'C', '1', '\0'};
constexpr uint32_t kAddDummyPrefix = 1u << 0;
constexpr uint32_t kRemoveExtraWhitespaces = 1u << 1;
constexpr uint32_t kEscapeWhitespaces = 1u << 2;
constexpr uint32_t kByteFallback = 1u << 3;
constexpr char kSpaceSymbol[] = "\xE2\x96\x81";  // ▁

inline int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

bool read_exact(std::FILE *f, void *dst, size_t n) { return std::fread(dst, 1, n, f) == n; }

}  // namespace

bool SentencePiece::load(const std::string &path, std::string *err) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "打不开词表: " + path;
        return false;
    }
    char magic[8];
    uint32_t n_pieces = 0, unk = 0, bos = 0, eos = 0, pad = 0;
    if (!read_exact(f, magic, 8) || std::memcmp(magic, kMagic, 8) != 0 ||
        !read_exact(f, &n_pieces, 4) || !read_exact(f, &unk, 4) || !read_exact(f, &bos, 4) ||
        !read_exact(f, &eos, 4) || !read_exact(f, &pad, 4) || !read_exact(f, &flags_, 4)) {
        std::fclose(f);
        if (err) *err = "词表头不对: " + path;
        return false;
    }
    unk_ = static_cast<int>(unk);
    bos_ = static_cast<int>(bos);
    eos_ = static_cast<int>(eos);
    pad_ = static_cast<int>(pad);

    uint32_t trie_bytes = 0;
    if (!read_exact(f, &trie_bytes, 4)) {
        std::fclose(f);
        if (err) *err = "词表缺少 trie 长度";
        return false;
    }
    trie_.resize(trie_bytes / 4);
    if (trie_bytes && !read_exact(f, trie_.data(), trie_bytes)) {
        std::fclose(f);
        if (err) *err = "词表 trie 被截断";
        return false;
    }
    uint32_t blob_bytes = 0;
    if (!read_exact(f, &blob_bytes, 4)) {
        std::fclose(f);
        if (err) *err = "词表缺少 blob 长度";
        return false;
    }
    blob_.resize(blob_bytes);
    if (blob_bytes && !read_exact(f, blob_.data(), blob_bytes)) {
        std::fclose(f);
        if (err) *err = "词表 blob 被截断";
        return false;
    }

    pieces_.resize(n_pieces);
    for (uint32_t i = 0; i < n_pieces; ++i) {
        uint8_t type = 0;
        float score = 0.0f;
        uint16_t len = 0;
        if (!read_exact(f, &type, 1) || !read_exact(f, &score, 4) || !read_exact(f, &len, 2)) {
            std::fclose(f);
            if (err) *err = "词表 pieces 被截断";
            return false;
        }
        Piece &p = pieces_[i];
        p.type = type;
        p.score = score;
        p.bytes.resize(len);
        if (len && !read_exact(f, p.bytes.data(), len)) {
            std::fclose(f);
            if (err) *err = "词表 piece 被截断";
            return false;
        }
    }
    std::fclose(f);

    by_bytes_.reserve(pieces_.size() * 2);
    byte_pieces_.fill(-1);
    for (size_t i = 0; i < pieces_.size(); ++i) {
        const Piece &p = pieces_[i];
        // piece 类型：0=normal 4=user_defined 6=byte 参与合并；2=unk 3=control 5=unused 不参与
        if (p.type == 2 || p.type == 3 || p.type == 5) continue;
        by_bytes_.emplace(p.bytes, static_cast<int32_t>(i));
        if (p.type == 6 && p.bytes.size() == 6 && p.bytes.compare(0, 3, "<0x") == 0) {
            const int hi = p.bytes[3], lo = p.bytes[4];
            auto hex = [](int c) {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                return -1;
            };
            const int v = (hex(hi) << 4) | hex(lo);
            if (v >= 0 && v < 256) byte_pieces_[static_cast<size_t>(v)] = static_cast<int32_t>(i);
        }
    }
    return true;
}

bool SentencePiece::prefix_match(const char *key, int len, int *matched, int *value) const {
    if (trie_.empty()) return false;
    uint32_t node = 0;
    int best_len = 0;
    int best_val = 0;
    for (int i = 0; i < len; ++i) {
        const uint32_t b = static_cast<uint8_t>(key[i]);
        const uint32_t unit = trie_[node];
        const uint32_t offset = (unit >> 10) << ((unit & (1u << 9)) >> 6);
        node ^= offset ^ b;
        if (node >= trie_.size()) break;
        const uint32_t u = trie_[node];
        if ((u & ((1u << 31) | 0xFFu)) != b) break;
        if ((u >> 8) & 1u) {
            const uint32_t leaf_off = (u >> 10) << ((u & (1u << 9)) >> 6);
            const uint32_t leaf = node ^ leaf_off;
            if (leaf < trie_.size()) {
                best_val = static_cast<int>(trie_[leaf] & ((1u << 31) - 1));
                best_len = i + 1;
            }
        }
    }
    if (best_len == 0) return false;
    *matched = best_len;
    *value = best_val;
    return true;
}

std::string SentencePiece::normalize(const std::string &text) const {
    const bool add_dummy = (flags_ & kAddDummyPrefix) != 0;
    const bool collapse = (flags_ & kRemoveExtraWhitespaces) != 0;
    const bool escape = (flags_ & kEscapeWhitespaces) != 0;

    std::string out;
    out.reserve(text.size() + 8);
    const char *src = text.data();
    const int n = static_cast<int>(text.size());
    int i = 0;
    if (collapse) {
        while (i < n && src[i] == ' ') ++i;
    }
    while (i < n) {
        int matched = 0, value = 0;
        if (prefix_match(src + i, n - i, &matched, &value) && value >= 0 &&
            static_cast<size_t>(value) < blob_.size()) {
            out += blob_.c_str() + value;  // blob 里是以 \0 结尾的替换串（可能是空串 = 删除）
            i += matched;
            continue;
        }
        const int len = std::min(utf8_len(static_cast<unsigned char>(src[i])), n - i);
        out.append(src + i, static_cast<size_t>(len));
        i += len;
    }
    if (collapse) {
        // 折叠连续空格并去掉尾部空格（与 sentencepiece 的顺序等价：映射后空格就是空格）
        std::string squeezed;
        squeezed.reserve(out.size());
        for (char c : out) {
            if (c == ' ' && !squeezed.empty() && squeezed.back() == ' ') continue;
            squeezed.push_back(c);
        }
        while (!squeezed.empty() && squeezed.back() == ' ') squeezed.pop_back();
        out.swap(squeezed);
    }
    if (escape) {
        std::string escaped;
        escaped.reserve(out.size() + 8);
        for (char c : out) {
            if (c == ' ') {
                escaped += kSpaceSymbol;
            } else {
                escaped.push_back(c);
            }
        }
        out.swap(escaped);
    }
    if (add_dummy) out.insert(0, kSpaceSymbol);
    return out;
}

std::vector<int32_t> SentencePiece::encode(const std::string &text) const {
    std::vector<int32_t> ids;
    const std::string norm = normalize(text);
    if (norm.empty()) return ids;

    std::vector<std::string> syms;
    syms.reserve(norm.size());
    for (size_t i = 0; i < norm.size();) {
        const int len = std::min(utf8_len(static_cast<unsigned char>(norm[i])), static_cast<int>(norm.size() - i));
        syms.emplace_back(norm, i, static_cast<size_t>(len));
        i += static_cast<size_t>(len);
    }

    // BPE：每次合并「分数最高的相邻对」，并列时取最左（与 sentencepiece 的优先队列等价）
    while (syms.size() > 1) {
        float best = -std::numeric_limits<float>::infinity();
        size_t best_i = syms.size();
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            std::string joined = syms[i] + syms[i + 1];
            auto it = by_bytes_.find(joined);
            if (it == by_bytes_.end()) continue;
            const float score = pieces_[static_cast<size_t>(it->second)].score;
            if (score > best) {
                best = score;
                best_i = i;
            }
        }
        if (best_i == syms.size()) break;
        syms[best_i] += syms[best_i + 1];
        syms.erase(syms.begin() + static_cast<long>(best_i) + 1);
    }

    ids.reserve(syms.size());
    for (const std::string &s : syms) {
        auto it = by_bytes_.find(s);
        if (it != by_bytes_.end()) {
            ids.push_back(it->second);
            continue;
        }
        if (flags_ & kByteFallback) {
            for (unsigned char c : s) ids.push_back(byte_pieces_[c]);
        } else {
            ids.push_back(unk_);
        }
    }
    return ids;
}

std::string SentencePiece::piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) return {};
    return pieces_[static_cast<size_t>(id)].bytes;
}

std::string SentencePiece::decode(const std::vector<int32_t> &ids) const {
    std::string text;
    for (int32_t id : ids) {
        const std::string p = piece(id);
        if (p.size() >= 3 && std::memcmp(p.data(), kSpaceSymbol, 3) == 0) {
            text.push_back(' ');
            text.append(p, 3, std::string::npos);
        } else {
            text += p;
        }
    }
    return text;
}

}  // namespace moss
