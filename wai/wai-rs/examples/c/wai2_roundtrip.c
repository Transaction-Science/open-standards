/*
 * WAI2 multi-rendition round-trip via the C ABI.
 *
 * Build:
 *   cargo build --release
 *   cc -I include examples/c/wai2_roundtrip.c \
 *      target/release/libwai.dylib -o /tmp/wai2_rt   # macOS
 *   cc -I include examples/c/wai2_roundtrip.c \
 *      target/release/libwai.so -ldl -lpthread -lm -o /tmp/wai2_rt   # Linux
 *
 * Run:
 *   /tmp/wai2_rt
 *
 * Exercises wai_envelope_pack_multi -> detect_version -> unpack_multi.
 * Verifies the rendition table JSON describes the payload block correctly.
 */

#include "wai.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAIL(msg) do { \
    fprintf(stderr, "FAIL: %s (last_error: %s)\n", (msg), \
            wai_last_error() ? wai_last_error() : "(none)"); \
    return 1; \
} while (0)

int main(void) {
    /* Two synthetic renditions of the "same content", with distinct
     * payloads so the round-trip is easy to verify byte-for-byte. */
    const uint8_t a_payload[] = { 0xEC, 0x01, 0x02, 0x03, 0xAA };   /* "encodec tokens" */
    const uint8_t b_payload[] = { 0x4F, 0x70, 0x75, 0x73 };          /* "Opus" magic-ish */

    const char *manifest =
        "{"
            "\"wai\":\"1.1\","
            "\"media\":\"audio\","
            "\"intent\":\"replicate\","
            "\"renditions\":["
                "{\"capability\":\"wai.neural.encodec32\",\"kind\":\"encodec_tokens\"},"
                "{\"capability\":\"wai.audio.opus\",\"kind\":\"opus\"}"
            "],"
            "\"target\":{\"sr\":48000,\"dur\":5.12}"
        "}";

    const uint8_t *payloads[2] = { a_payload, b_payload };
    uintptr_t      lens[2]     = { sizeof a_payload, sizeof b_payload };

    /* ---- pack -------------------------------------------------- */
    struct wai_buffer_t envelope = {0};
    int rc = wai_envelope_pack_multi(manifest, payloads, lens, 2, &envelope);
    if (rc != 0) FAIL("pack_multi");
    printf("packed WAI2 envelope: %zu bytes\n", (size_t)envelope.len);

    /* ---- detect_version --------------------------------------- */
    int v = wai_envelope_detect_version(envelope.data, envelope.len);
    if (v != 2) FAIL("detect_version != 2");
    printf("detect_version: %d\n", v);

    /* ---- unpack ----------------------------------------------- */
    struct wai_buffer_t mout = {0}, table = {0}, block = {0};
    rc = wai_envelope_unpack_multi(envelope.data, envelope.len,
                                   &mout, &table, &block);
    if (rc != 0) FAIL("unpack_multi");

    printf("manifest JSON (%zu B): %.*s\n",
           (size_t)mout.len, (int)mout.len, (const char *)mout.data);
    printf("rendition table (%zu B): %.*s\n",
           (size_t)table.len, (int)table.len, (const char *)table.data);
    printf("payload block: %zu B\n", (size_t)block.len);

    /* The block layout is offset-described by the table JSON. With two
     * known payloads back-to-back we can verify the bytes directly. */
    size_t want = sizeof a_payload + sizeof b_payload;
    if (block.len != want) {
        fprintf(stderr, "block.len %zu != %zu\n", (size_t)block.len, want);
        return 1;
    }
    if (memcmp(block.data,                     a_payload, sizeof a_payload) != 0)
        FAIL("payload 0 mismatch");
    if (memcmp(block.data + sizeof a_payload,  b_payload, sizeof b_payload) != 0)
        FAIL("payload 1 mismatch");

    /* ---- free everything we got back -------------------------- */
    wai_buffer_free(envelope);
    wai_buffer_free(mout);
    wai_buffer_free(table);
    wai_buffer_free(block);

    puts("ok");
    return 0;
}
