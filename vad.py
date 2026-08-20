import time

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad


SAMPLE_RATE = 16000
CHANNELS = 1

# Silero supports 16 kHz.
# 512 samples = 32 ms at 16 kHz.
CHUNK_SAMPLES = 512


print("Loading Silero VAD...", flush=True)

# VAD is intentionally CPU-only.
torch.set_num_threads(1)

model = load_silero_vad()

print("VAD loaded.", flush=True)
print("Starting microphone...", flush=True)
print()


# ------------------------------------------------------------
# VAD state
# ------------------------------------------------------------

speech_active = False
speech_start_time = None

audio_time = 0.0


def process_audio(audio):
    global speech_active
    global speech_start_time
    global audio_time

    # sounddevice gives us float32 numpy arrays.
    # Silero expects a torch tensor.
    tensor = torch.from_numpy(audio).float()

    # Get speech probability.
    probability = model(tensor, SAMPLE_RATE).item()

    chunk_start = audio_time
    chunk_end = audio_time + len(audio) / SAMPLE_RATE

    audio_time = chunk_end

    # Basic threshold for now.
    is_speech = probability >= 0.5

    # Print probability so we can tune this later.
    print(
        f"\r"
        f"time={chunk_end:7.2f}s "
        f"prob={probability:.3f} "
        f"{'SPEECH ' if is_speech else 'silence'}",
        end="",
        flush=True,
    )

    if is_speech and not speech_active:
        speech_active = True
        speech_start_time = chunk_start

        print()
        print(
            f"[VAD] SPEECH START "
            f"at {speech_start_time:.2f}s",
            flush=True,
        )

    elif not is_speech and speech_active:
        speech_active = False

        print()
        print(
            f"[VAD] SPEECH END "
            f"at {chunk_start:.2f}s",
            flush=True,
        )


# ------------------------------------------------------------
# Microphone callback
# ------------------------------------------------------------

def audio_callback(indata, frames, time_info, status):
    if status:
        print(
            f"\n[AUDIO] {status}",
            flush=True,
        )

    # Convert (N, 1) -> (N,)
    audio = indata[:, 0].copy()

    process_audio(audio)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
    ):
        print("Microphone is running.")
        print("Speak normally. Ctrl+C to stop.")
        print()

        while True:
            time.sleep(1)

except KeyboardInterrupt:
    print()
    print()
    print("Stopped.")

except Exception as e:
    print()
    print()
    print("ERROR:")
    print(repr(e))
