/*
 * Native neural decode through the C ABI.
 *
 * Builds against libwai.dylib (with the `neural` feature) and decodes
 * one each of the audio, image, and video sample WAI envelopes.
 *
 * Build (from repo root):
 *   cargo build --release --features neural --manifest-path wai-rs/Cargo.toml
 *   cc -I wai-rs/include wai-rs/examples/c/neural_decode.c \
 *      wai-rs/target/release/libwai.dylib -o /tmp/wai_neural
 *
 * Run:
 *   DYLD_LIBRARY_PATH=wai-rs/target/release /tmp/wai_neural
 */

#include "wai.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPO  ".."
#define M(n)  REPO "/wai-web/demo/models/" n "/decoder.onnx"
#define S(n)  REPO "/wai-web/demo/samples/" n

#define CHECK(call, msg) do { \
    int rc = (call); \
    if (rc != 0) { fprintf(stderr, "FAIL %s: rc=%d\n", (msg), rc); return rc; } \
} while (0)

static int load_file(const char *path, uint8_t **out, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    *out = (uint8_t *)malloc((size_t)n);
    *out_len = (size_t)fread(*out, 1, (size_t)n, f);
    fclose(f);
    return 0;
}

int main(void) {
    /* ---- audio: wai.neural.encodec32 -------------------------- */
    {
        uint8_t *env; size_t env_len;
        if (load_file(S("glass.encodec.wai"), &env, &env_len) != 0) return 1;
        struct wai_buffer_t samples = {0};
        uint32_t sr = 0;
        CHECK(wai_neural_decode_audio(env, env_len, M("encodec_32khz"),
                                       &samples, &sr),
              "encodec32 decode");
        size_t n = samples.len / sizeof(float);
        const float *s = (const float *)samples.data;
        float peak = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float a = s[i] < 0 ? -s[i] : s[i];
            if (a > peak) peak = a;
        }
        printf("encodec32     : %zu samples @ %u Hz, peak %.3f\n", n, sr, peak);
        wai_buffer_free(samples);
        free(env);
    }

    /* ---- image: wai.neural.bmshj2018 --------------------------- */
    {
        uint8_t *env; size_t env_len;
        if (load_file(S("kodim23.bmshj2018.wai"), &env, &env_len) != 0) return 1;
        struct wai_buffer_t rgb = {0};
        uint32_t w = 0, h = 0;
        CHECK(wai_neural_decode_image(env, env_len, M("bmshj2018"), &rgb, &w, &h),
              "bmshj2018 decode");
        printf("bmshj2018     : %ux%u RGB, %zu bytes\n", w, h, (size_t)rgb.len);
        wai_buffer_free(rgb);
        free(env);
    }

    /* ---- video: wai.neural.video_bmshj2018 ---------------------- */
    {
        uint8_t *env; size_t env_len;
        if (load_file(S("test.video.wai"), &env, &env_len) != 0) return 1;
        struct wai_buffer_t frames = {0};
        uint32_t n_frames = 0, w = 0, h = 0, fps_x_1000 = 0;
        CHECK(wai_neural_decode_video(env, env_len, M("bmshj2018"),
                                       &frames, &n_frames, &w, &h, &fps_x_1000),
              "video_bmshj2018 decode");
        printf("video_bmshj   : %u frames %ux%u @ %.2f fps, %zu bytes\n",
               n_frames, w, h, fps_x_1000 / 1000.0, (size_t)frames.len);
        wai_buffer_free(frames);
        free(env);
    }

    puts("ok");
    return 0;
}
