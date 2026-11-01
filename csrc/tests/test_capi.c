/* C API 冒烟测试：用纯 C 编译，链接 moss_tts.lib，跑一次合成。
 * 这个文件同时也是给使用者的最小示例。
 */
#include <stdio.h>
#include <string.h>

#include "moss/tts.h"

static int g_total_frames = 0;
static int g_peak = 0;

static void on_pcm(void *user, const int16_t *pcm, int frames, int sample_rate) {
    (void)user;
    (void)sample_rate;
    g_total_frames += frames;
    for (int i = 0; i < frames * 2; ++i) {
        const int v = pcm[i] < 0 ? -pcm[i] : pcm[i];
        if (v > g_peak) g_peak = v;
    }
}

int main(int argc, char **argv) {
    const char *models = argc > 1 ? argv[1] : "models";
    char llm[512], dec[512], enc[512], vocab[512], voices[512], cache[512];
    snprintf(llm, sizeof(llm), "%s/llm_q8x4.moss", models);
    snprintf(dec, sizeof(dec), "%s/decoder_f16.moss", models);
    snprintf(enc, sizeof(enc), "%s/encoder_f16.moss", models);
    snprintf(vocab, sizeof(vocab), "%s/tokenizer.mossvocab", models);
    snprintf(voices, sizeof(voices), "%s/voices", models);
    snprintf(cache, sizeof(cache), "%s/cache", models);

    printf("version: %s\n", moss_version());
    moss_engine *engine = moss_create(llm, dec, enc, vocab, voices, cache, 4, 6);
    if (!engine) {
        printf("moss_create failed\n");
        return 1;
    }
    if (moss_set_voice(engine, "Junhao") != 0) {
        printf("set_voice failed: %s\n", moss_last_error(engine));
        moss_destroy(engine);
        return 1;
    }
    printf("tokens: %d\n", moss_text_tokens(engine, "\xe4\xbd\xa0\xe5\xa5\xbd")); /* 你好 */
    if (moss_speak(engine, "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xe3\x80\x82", on_pcm,
                   NULL, 0, 0, 375, 8) != 0) {
        printf("speak failed: %s\n", moss_last_error(engine));
        moss_destroy(engine);
        return 1;
    }
    printf("frames: %d peak: %d\n", g_total_frames, g_peak);
    moss_destroy(engine);
    return (g_total_frames > 0 && g_peak > 100) ? 0 : 2;
}
