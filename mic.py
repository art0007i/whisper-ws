import sounddevice as sd
from scipy.io.wavfile import write


SAMPLE_RATE = 16000
CHANNELS = 1
SECONDS = 10
OUTPUT = "mic_test.wav"


print("Available audio devices:")
print(sd.query_devices())

print()
print(f"Recording for {SECONDS} seconds...")
print("Speak into your microphone!")
print()

try:
    audio = sd.rec(
        int(SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )

    sd.wait()

    write(
        OUTPUT,
        SAMPLE_RATE,
        audio,
    )

    print()
    print(f"Saved recording to: {OUTPUT}")

except Exception as e:
    print()
    print("MICROPHONE ERROR:")
    print(repr(e))
