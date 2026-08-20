import ctypes
import platform
from pathlib import Path


class WhisperGPU:
    def __init__(self, model_path: str):
        self.model_path = str(Path(model_path).resolve())

        # CMake places the wrapper directly beside this file.
        here = Path(__file__).resolve().parent

        if platform.system() == "Windows":
            lib_name = "whisper_wrapper.dll"
        else:
            lib_name = "libwhisper_wrapper.so"

        lib_path = here / lib_name

        if not lib_path.exists():
            raise FileNotFoundError(
                f"Whisper wrapper not found: {lib_path}"
            )

        self.lib = ctypes.CDLL(str(lib_path))

        # WhisperHandle * whisper_create(const char *)
        self.lib.whisper_create.argtypes = [ctypes.c_char_p]
        self.lib.whisper_create.restype = ctypes.c_void_p

        # void whisper_destroy(WhisperHandle *)
        self.lib.whisper_destroy.argtypes = [ctypes.c_void_p]
        self.lib.whisper_destroy.restype = None

        # int whisper_transcribe(
        #     WhisperHandle *,
        #     const float *,
        #     int,
        #     char *,
        #     int
        # )
        self.lib.whisper_transcribe.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.whisper_transcribe.restype = ctypes.c_int

        self.handle = self.lib.whisper_create(
            self.model_path.encode("utf-8")
        )

        if not self.handle:
            raise RuntimeError(
                f"Failed to initialize whisper.cpp with model: "
                f"{self.model_path}"
            )

    def transcribe(self, audio):
        audio = audio.astype("float32", copy=False)

        output_size = 16384
        output = ctypes.create_string_buffer(output_size)

        result = self.lib.whisper_transcribe(
            self.handle,
            audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(audio),
            output,
            output_size,
        )

        if result != 0:
            raise RuntimeError(
                f"whisper_transcribe failed with error {result}"
            )

        return output.value.decode("utf-8", errors="replace").strip()

    def __del__(self):
        handle = getattr(self, "handle", None)

        if handle:
            try:
                self.lib.whisper_destroy(handle)
            except Exception:
                pass

            self.handle = None
