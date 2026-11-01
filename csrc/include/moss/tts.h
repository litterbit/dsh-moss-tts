// MOSS-TTS CPU 推理引擎的 C 接口（供插件 / 设备 / 网络播放器调用）。
//
//   moss_engine *e = moss_create("models/llm_q8x4.moss", "models/decoder_f16.moss",
//                                "models/encoder_f16.moss", "models/tokenizer.mossvocab",
//                                "models/voices", "cache", 4, 6);
//   moss_set_voice(e, "Junhao");            // 或 "voices/me.mossclone"
//   moss_speak(e, "你好，世界。", on_pcm, ud, 0, 0, 375, 8);
//   moss_destroy(e);
//
// 所有函数返回 0 表示成功，非 0 表示失败（用 moss_last_error 取原因）。
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moss_engine moss_engine;

// 交织 int16 PCM 回调，在 moss_speak 的调用线程上同步触发
typedef void (*moss_pcm_cb)(void *user, const int16_t *pcm, int frames, int sample_rate);

// encoder_path / voices_dir / cache_dir 可以为 NULL
moss_engine *moss_create(const char *llm_path, const char *decoder_path, const char *encoder_path,
                         const char *tokenizer_path, const char *voices_dir, const char *cache_dir,
                         int llm_threads, int decode_threads);
void moss_destroy(moss_engine *engine);

int moss_set_voice(moss_engine *engine, const char *name_or_profile);
int moss_speak(moss_engine *engine, const char *text, moss_pcm_cb on_pcm, void *user, uint64_t seed,
               int greedy, int max_frames, int chunk_frames);
int moss_clone(moss_engine *engine, const char *wav_path, const char *out_profile);
int moss_text_tokens(moss_engine *engine, const char *text);
const char *moss_last_error(moss_engine *engine);
const char *moss_version(void);

#ifdef __cplusplus
}
#endif
