import ctypes
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

if sys.platform.startswith("win"):
    lib_name = "whisper_wrapper.dll"
elif sys.platform == "darwin":
    lib_name = "libwhisper_wrapper.dylib"
else:
    lib_name = "libwhisper_wrapper.so"

lib = ctypes.CDLL(str(ROOT / lib_name))

class WhisperHandle(ctypes.Structure):
    pass

WhisperHandlePtr = ctypes.POINTER(WhisperHandle)


lib.whisper_create.argtypes = [
    ctypes.c_char_p,
]

lib.whisper_create.restype = WhisperHandlePtr


lib.whisper_destroy.argtypes = [
    WhisperHandlePtr,
]

lib.whisper_destroy.restype = None


lib.whisper_transcribe.argtypes = [
    WhisperHandlePtr,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
]

lib.whisper_transcribe.restype = ctypes.c_int


class WhisperGPU:
    def __init__(self, model_path):
        model_path = str(model_path).encode()

        print("Loading whisper.cpp model on Vulkan GPU...")

        self.ctx = lib.whisper_create(model_path)

        if not self.ctx:
            raise RuntimeError("Failed to initialize whisper.cpp")

        print("Whisper GPU model ready.")

    def transcribe(self, audio):
        """
        audio:
            numpy float32 array, mono, 16 kHz
        """

        import numpy as np

        audio = np.asarray(audio, dtype=np.float32)

        output = ctypes.create_string_buffer(16384)

        ptr = audio.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        ret = lib.whisper_transcribe(
            self.ctx,
            ptr,
            len(audio),
            output,
            len(output),
        )

        if ret != 0:
            raise RuntimeError(
                f"whisper_transcribe failed: {ret}"
            )

        return output.value.decode(
            "utf-8",
            errors="replace",
        ).strip()

    def close(self):
        if self.ctx:
            lib.whisper_destroy(self.ctx)
            self.ctx = None

    def __del__(self):
        self.close()
