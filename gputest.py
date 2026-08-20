import wave
import numpy as np

from whisper_gpu.whisper_gpu import WhisperGPU


with wave.open("output.wav", "rb") as f:
    assert f.getnchannels() == 1
    assert f.getsampwidth() == 2
    assert f.getframerate() == 16000

    pcm = f.readframes(f.getnframes())

audio = (
    np.frombuffer(pcm, dtype=np.int16)
    .astype(np.float32)
    / 32768.0
)

whisper = WhisperGPU("models/ggml-base.bin")

text = whisper.transcribe(audio)

print("RESULT:", text)

whisper.close()
