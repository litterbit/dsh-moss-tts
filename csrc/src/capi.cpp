#include "moss/tts.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "moss/engine.h"

using moss::EngineOptions;
using moss::SynthOptions;
using moss::SynthStats;
using moss::TtsEngine;

struct moss_engine {
    TtsEngine engine;
    std::string error;
};

namespace {

const char *kVersion = "moss-tts-cpu 0.4.0";

inline int16_t to_pcm16(float v) {
    const float clamped = std::max(-1.0f, std::min(1.0f, v));
    return static_cast<int16_t>(std::lrintf(clamped * 32767.0f));
}

}  // namespace

extern "C" {

moss_engine *moss_create(const char *llm_path, const char *decoder_path, const char *encoder_path,
                         const char *tokenizer_path, const char *voices_dir, const char *cache_dir,
                         int llm_threads, int decode_threads) {
    auto *handle = new moss_engine();
    EngineOptions options;
    options.llm_path = llm_path ? llm_path : "";
    options.decoder_path = decoder_path ? decoder_path : "";
    options.encoder_path = encoder_path ? encoder_path : "";
    options.tokenizer_path = tokenizer_path ? tokenizer_path : "";
    options.voices_dir = voices_dir ? voices_dir : "";
    options.cache_dir = cache_dir ? cache_dir : "";
    options.llm_threads = llm_threads > 0 ? llm_threads : 4;
    options.decode_threads = decode_threads > 0 ? decode_threads : options.llm_threads;
    if (!handle->engine.load(options, &handle->error)) {
        delete handle;
        return nullptr;
    }
    return handle;
}

void moss_destroy(moss_engine *engine) { delete engine; }

int moss_set_voice(moss_engine *engine, const char *name_or_profile) {
    if (!engine || !name_or_profile) return 1;
    return engine->engine.set_voice(name_or_profile, &engine->error) ? 0 : 1;
}

int moss_speak(moss_engine *engine, const char *text, moss_pcm_cb on_pcm, void *user, uint64_t seed,
               int greedy, int max_frames, int chunk_frames) {
    if (!engine || !text) return 1;
    SynthOptions options;
    options.seed = seed;
    options.greedy = greedy != 0;
    options.max_frames = max_frames > 0 ? max_frames : 375;
    options.chunk_frames = chunk_frames > 0 ? chunk_frames : 8;
    std::vector<int16_t> pcm;
    options.on_audio = [&](const float *data, int frames) {
        pcm.resize(static_cast<size_t>(frames) * 2);
        for (int i = 0; i < frames * 2; ++i) pcm[static_cast<size_t>(i)] = to_pcm16(data[i]);
        if (on_pcm) on_pcm(user, pcm.data(), frames, 48000);
    };
    SynthStats stats;
    return engine->engine.synthesize(text, options, &stats, &engine->error) ? 0 : 1;
}

int moss_clone(moss_engine *engine, const char *wav_path, const char *out_profile) {
    if (!engine || !wav_path || !out_profile) return 1;
    return engine->engine.clone_audio(wav_path, out_profile, &engine->error) ? 0 : 1;
}

int moss_text_tokens(moss_engine *engine, const char *text) {
    if (!engine || !text) return -1;
    return static_cast<int>(engine->engine.text_to_ids(text).size());
}

const char *moss_last_error(moss_engine *engine) { return engine ? engine->error.c_str() : "null engine"; }

const char *moss_version(void) { return kVersion; }

}  // extern "C"
