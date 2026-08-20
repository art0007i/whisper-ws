#include "whisper.h"

#include <cstdio>
#include <string>
#include <vector>

extern "C" {

struct WhisperHandle {
    whisper_context * ctx;
};

WhisperHandle * whisper_create(const char * model_path) {
    whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu = true;
    cparams.flash_attn = true;
    cparams.gpu_device = 0;

    WhisperHandle * h = new WhisperHandle{};
    h->ctx = whisper_init_from_file_with_params(model_path, cparams);

    if (!h->ctx) {
        delete h;
        return nullptr;
    }

    return h;
}

void whisper_destroy(WhisperHandle * h) {
    if (!h) {
        return;
    }

    if (h->ctx) {
        whisper_free(h->ctx);
    }

    delete h;
}

int whisper_transcribe(
    WhisperHandle * h,
    const float * audio,
    int n_samples,
    const char * language,
    char * output,
    int output_size,
    char * lang_out,
    int lang_out_size
) {
    if (!h || !h->ctx || !audio || !output || output_size <= 0) {
        return -1;
    }

    whisper_full_params params =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;

    params.translate = false;
    params.no_context = true;

    params.language = language;

    // Four CPU threads for the parts that remain on CPU.
    params.n_threads = 4;

    if (whisper_full(h->ctx, params, audio, n_samples) != 0) {
        return -2;
    }

    std::string result;

    const int n_segments = whisper_full_n_segments(h->ctx);

    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(h->ctx, i);

        if (text) {
            result += text;
        }
    }

    if ((int)result.size() >= output_size) {
        result.resize(output_size - 1);
    }

    std::snprintf(
        output,
        output_size,
        "%s",
        result.c_str()
    );

    if (lang_out && lang_out_size > 0) {
        const int lang_id = whisper_full_lang_id(h->ctx);
        const char * lang = whisper_lang_str(lang_id);
	if (lang) std::snprintf(lang_out, lang_out_size, "%s", lang);
    }

    return 0;
}

}
