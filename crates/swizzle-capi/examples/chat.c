/*
 * Asks a local model one question and streams the answer.
 *
 *   cl /I include examples\chat.c <target>\release\swizzle_capi.dll.lib
 *   gcc -I include examples/chat.c -L <target>/release -lswizzle_capi
 *
 *   chat model.gguf "Name three rivers."
 */
#include <stdio.h>

#include "swizzle.h"

static bool print_text(const char *text, size_t len, void *user_data) {
    (void)user_data;
    fwrite(text, 1, len, stdout);
    fflush(stdout);
    return true;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL PROMPT\n", argv[0]);
        return 2;
    }
    swizzle_model *model = swizzle_model_load(argv[1]);
    if (!model) {
        fprintf(stderr, "swizzle: %s\n", swizzle_last_error());
        return 1;
    }
    swizzle_session *session = swizzle_session_new(model, NULL);
    if (!session) {
        fprintf(stderr, "swizzle: %s\n", swizzle_last_error());
        swizzle_model_free(model);
        return 1;
    }
    swizzle_params params = swizzle_params_default();
    params.max_tokens = 128;
    int64_t tokens = swizzle_session_generate(session, argv[2], &params, print_text, NULL);
    if (tokens < 0) {
        fprintf(stderr, "swizzle: %s\n", swizzle_last_error());
    } else {
        fprintf(stderr, "\n[%lld tokens, swizzle %s]\n", (long long)tokens, swizzle_version());
    }
    swizzle_session_free(session);
    swizzle_model_free(model);
    return tokens < 0;
}
