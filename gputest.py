import time

import numpy as np
import soundfile as sf

from whisper_gpu.whisper_gpu import WhisperGPU


MODEL = "models/ggml-base.bin"
AUDIO = "output.wav"


print("=" * 60)
print("whisper.cpp Vulkan GPU test")
print("=" * 60)

print(f"Model: {MODEL}")
print(f"Audio: {AUDIO}")

print()
print("Loading whisper.cpp model on Vulkan GPU...")

whisper = WhisperGPU(MODEL)

print("Model loaded.")
print()

print(f"Reading {AUDIO}...")

audio, sample_rate = sf.read(
    AUDIO,
    dtype="float32",
)

# Convert stereo -> mono if necessary.
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

if sample_rate != 16000:
    raise RuntimeError(
        f"Expected 16 kHz audio, got {sample_rate} Hz"
    )

print(
    f"Audio: {len(audio)} samples "
    f"({len(audio) / sample_rate:.2f}s)"
)

print()
print("Transcribing...")

start = time.perf_counter()

text = whisper.transcribe(audio)

elapsed = time.perf_counter() - start

print()
print("-" * 60)
print("RESULT")
print("-" * 60)
print(text)
print("-" * 60)

print(
    f"Transcription time: {elapsed:.3f}s"
)

if elapsed > 0:
    print(
        f"Realtime factor: "
        f"{(len(audio) / sample_rate) / elapsed:.2f}x"
    )

print()
print("GPU test successful.")
