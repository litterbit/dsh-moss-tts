// SPM 分词对拍：读 tests/data/spm_corpus.txt + spm_golden.txt，逐句比较 ids。
//
//   moss_spmtest --vocab weights/tokenizer.mossvocab \
//                --corpus tests/data/spm_corpus.txt --golden tests/data/spm_golden.txt
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "moss/spm.h"

using namespace moss;

namespace {

std::vector<std::string> read_lines(const std::string &path) {
    std::vector<std::string> lines;
    std::ifstream in(path, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
    }
    return lines;
}

std::vector<int32_t> parse_ids(const std::string &line) {
    std::vector<int32_t> ids;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && line[i] == ' ') ++i;
        if (i >= line.size()) break;
        int value = 0;
        while (i < line.size() && line[i] >= '0' && line[i] <= '9') {
            value = value * 10 + (line[i] - '0');
            ++i;
        }
        ids.push_back(value);
    }
    return ids;
}

}  // namespace

int main(int argc, char **argv) {
    std::string vocab, corpus_path, golden_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--vocab") vocab = next();
        else if (a == "--corpus") corpus_path = next();
        else if (a == "--golden") golden_path = next();
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            return 1;
        }
    }
    if (vocab.empty() || corpus_path.empty() || golden_path.empty()) {
        std::fprintf(stderr, "用法: moss_spmtest --vocab <x.mossvocab> --corpus <a.txt> --golden <b.txt>\n");
        return 1;
    }

    SentencePiece spm;
    std::string err;
    if (!spm.load(vocab, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    // 语料按行读；golden 也按行（语料里的 \n 已被写成真正的换行，所以用 \x1f 之外的方式没法对齐）
    const std::vector<std::string> corpus = read_lines(corpus_path);
    const std::vector<std::string> golden = read_lines(golden_path);
    if (corpus.size() != golden.size()) {
        std::fprintf(stderr, "语料 %zu 行与 golden %zu 行不匹配（注意语料里不能有裸换行）\n", corpus.size(),
                     golden.size());
        return 1;
    }

    int exact = 0;
    int total_ids = 0;
    int matched_ids = 0;
    for (size_t i = 0; i < corpus.size(); ++i) {
        const std::vector<int32_t> got = spm.encode(corpus[i]);
        const std::vector<int32_t> want = parse_ids(golden[i]);
        bool same = got.size() == want.size();
        if (same) {
            for (size_t k = 0; k < got.size(); ++k) {
                if (got[k] != want[k]) {
                    same = false;
                    break;
                }
            }
        }
        if (same) ++exact;
        total_ids += static_cast<int>(want.size());
        for (size_t k = 0; k < got.size() && k < want.size(); ++k) {
            if (got[k] == want[k]) ++matched_ids;
        }
        if (!same) {
            std::printf("[diff] #%zu %s\n", i, corpus[i].c_str());
            std::printf("   want:");
            for (int32_t v : want) std::printf(" %d", v);
            std::printf("\n   got :");
            for (int32_t v : got) std::printf(" %d", v);
            std::printf("\n   want-text: %s\n", spm.decode(want).c_str());
            std::printf("   got-text : %s\n", spm.decode(got).c_str());
        }
    }
    std::printf("{\n  \"sentences\": %zu,\n  \"exact\": %d,\n  \"exact_rate\": %.4f,\n  \"id_match_rate\": %.4f\n}\n",
                corpus.size(), exact, static_cast<double>(exact) / static_cast<double>(corpus.size()),
                total_ids ? static_cast<double>(matched_ids) / total_ids : 0.0);
    return exact == static_cast<int>(corpus.size()) ? 0 : 2;
}
