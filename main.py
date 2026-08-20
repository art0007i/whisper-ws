import asyncio
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from silero_vad import load_silero_vad, VADIterator
import uvicorn

from whisper_gpu.whisper_gpu import WhisperGPU


# ============================================================
# Configuration
# ============================================================

HOST = "0.0.0.0"
PORT = 6912

SAMPLE_RATE = 16000
CHANNELS = 1

VAD_CHUNK_SAMPLES = 512

MIN_SILENCE_MS = 600
SPEECH_PAD_MS = 200

# ------------------------------------------------------------
# Rolling transcription
# ------------------------------------------------------------

# How often an interim transcription is produced.
INTERIM_INTERVAL = 0.75

# Only the newest portion of the utterance is sent to Whisper
# for interim recognition.
INTERIM_WINDOW_SECONDS = 4.0

# Don't attempt an interim until at least this much audio exists.
INTERIM_MIN_AUDIO_SECONDS = 0.8

# ------------------------------------------------------------
# Whisper
# ------------------------------------------------------------

WHISPER_MODEL = "models/ggml-base.bin"


# ============================================================
# Application state
# ============================================================

app = FastAPI()

audio_queue = queue.Queue()

clients = set()

event_loop = None

whisper_lock = threading.Lock()


# ============================================================
# Load models
# ============================================================

print("Loading Whisper.cpp Vulkan GPU model...", flush=True)

whisper = WhisperGPU(WHISPER_MODEL)

print("Whisper.cpp GPU model loaded.", flush=True)


print("Loading Silero VAD...", flush=True)

vad_model = load_silero_vad()

print("VAD loaded.", flush=True)


# ============================================================
# WebSocket protocol
# ============================================================

def make_message(
    message_type,
    language,
    start,
    end,
    text,
):
    return (
        f"type={message_type}\n"
        f"lang={language}\n"
        f"start={start:.2f}\n"
        f"end={end:.2f}\n"
        f"text={text}\n"
    )


async def broadcast(message):
    if not clients:
        return

    dead = []

    for client in list(clients):
        try:
            await client.send_text(message)
        except Exception:
            dead.append(client)

    for client in dead:
        clients.discard(client)


def broadcast_from_thread(message):
    if event_loop is None:
        return

    asyncio.run_coroutine_threadsafe(
        broadcast(message),
        event_loop,
    )


# ============================================================
# Microphone
# ============================================================

def microphone_callback(indata, frames, time_info, status):
    if status:
        print(f"\n[AUDIO] {status}", flush=True)

    audio = indata[:, 0].copy()

    audio_queue.put(audio)


def microphone_worker():
    print("Starting microphone...", flush=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=VAD_CHUNK_SAMPLES,
            callback=microphone_callback,
        ):
            print("Microphone running.", flush=True)

            while True:
                time.sleep(1)

    except Exception as e:
        print(
            f"\n[MICROPHONE ERROR] {e!r}",
            flush=True,
        )


# ============================================================
# Whisper
# ============================================================

def transcribe(audio):
    """
    Transcribe a float32 mono numpy array using whisper.cpp.
    """

    if len(audio) == 0:
        return ""

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    with whisper_lock:
        return whisper.transcribe(audio)


# ============================================================
# Speech processing
# ============================================================

def speech_worker():
    print("Speech/VAD worker started.", flush=True)

    vad = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )

    speech_audio = []

    speech_active = False

    speech_start_time = None

    last_interim_time = 0.0

    while True:
        audio = audio_queue.get()

        try:
            tensor = np.asarray(
                audio,
                dtype=np.float32,
            )

            import torch

            tensor_torch = torch.from_numpy(tensor)

            vad_event = vad(
                tensor_torch,
                return_seconds=False,
            )

            now = time.monotonic()

            # ------------------------------------------------
            # Speech start
            # ------------------------------------------------

            if (
                vad_event is not None
                and "start" in vad_event
            ):
                speech_active = True

                speech_start_time = (
                    vad_event["start"] / SAMPLE_RATE
                )

                print(
                    f"\n[VAD] SPEECH START "
                    f"{speech_start_time:.2f}s",
                    flush=True,
                )

                speech_audio = [
                    tensor.copy()
                ]

                last_interim_time = now

                continue

            # ------------------------------------------------
            # Collect speech
            # ------------------------------------------------

            if speech_active:
                speech_audio.append(
                    tensor.copy()
                )

            # ------------------------------------------------
            # Rolling interim transcription
            # ------------------------------------------------

            if speech_active:
                elapsed = (
                    now - last_interim_time
                )

                total_samples = sum(
                    len(x)
                    for x in speech_audio
                )

                total_seconds = (
                    total_samples / SAMPLE_RATE
                )

                if (
                    elapsed >= INTERIM_INTERVAL
                    and total_seconds >= INTERIM_MIN_AUDIO_SECONDS
                ):
                    do_interim_transcription(
                        speech_audio,
                        speech_start_time,
                    )

                    last_interim_time = now

            # ------------------------------------------------
            # Speech end
            # ------------------------------------------------

            if (
                vad_event is not None
                and "end" in vad_event
            ):
                if not speech_active:
                    continue

                end_time = (
                    vad_event["end"]
                    / SAMPLE_RATE
                )

                print(
                    f"\n[VAD] SPEECH END "
                    f"{end_time:.2f}s",
                    flush=True,
                )

                speech_active = False

                audio_data = np.concatenate(
                    speech_audio
                )

                speech_audio = []

                # Final transcription of the entire utterance.
                do_final_transcription(
                    audio_data,
                    speech_start_time,
                    end_time,
                )

                speech_start_time = None

                last_interim_time = 0.0

                vad.reset_states()

        except Exception as e:
            print(
                f"\n[SPEECH ERROR] {e!r}",
                flush=True,
            )


# ============================================================
# Interim transcription
# ============================================================

def do_interim_transcription(
    speech_audio,
    speech_start_time,
):
    try:
        total_samples = sum(
            len(x)
            for x in speech_audio
        )

        window_samples = int(
            INTERIM_WINDOW_SECONDS
            * SAMPLE_RATE
        )

        # Don't retranscribe the whole utterance.
        # Keep only the newest window.
        start_sample = max(
            0,
            total_samples - window_samples,
        )

        chunks = []

        position = 0

        for chunk in speech_audio:
            chunk_end = position + len(chunk)

            if chunk_end > start_sample:
                local_start = max(
                    0,
                    start_sample - position,
                )

                chunks.append(
                    chunk[local_start:]
                )

            position = chunk_end

        if not chunks:
            return

        audio = np.concatenate(chunks)

        text = transcribe(audio)

        if not text:
            return

        window_start = (
            start_sample / SAMPLE_RATE
        )

        window_end = (
            total_samples / SAMPLE_RATE
        )

        message = make_message(
            "interim",
            "en",
            speech_start_time + window_start,
            speech_start_time + window_end,
            text,
        )

        print(
            f"[INTERIM] {text}",
            flush=True,
        )

        broadcast_from_thread(message)

    except Exception as e:
        print(
            f"\n[INTERIM ERROR] {e!r}",
            flush=True,
        )


# ============================================================
# Final transcription
# ============================================================

def do_final_transcription(
    audio,
    speech_start_time,
    speech_end_time,
):
    try:
        text = transcribe(audio)

        if not text:
            return

        message = make_message(
            "final",
            "en",
            speech_start_time,
            speech_end_time,
            text,
        )

        print(
            f"[FINAL] {text}",
            flush=True,
        )

        broadcast_from_thread(message)

    except Exception as e:
        print(
            f"\n[FINAL ERROR] {e!r}",
            flush=True,
        )


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
):
    await websocket.accept()

    clients.add(websocket)

    print(
        f"\n[WS] Client connected "
        f"({len(clients)} clients)",
        flush=True,
    )

    try:
        while True:
            await websocket.receive()

    except WebSocketDisconnect:
        pass

    except Exception as e:
        print(
            f"\n[WS ERROR] {e!r}",
            flush=True,
        )

    finally:
        clients.discard(websocket)

        print(
            f"\n[WS] Client disconnected "
            f"({len(clients)} clients)",
            flush=True,
        )


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
async def startup():
    global event_loop

    event_loop = asyncio.get_running_loop()

    threading.Thread(
        target=microphone_worker,
        daemon=True,
    ).start()

    threading.Thread(
        target=speech_worker,
        daemon=True,
    ).start()

    print()
    print("=" * 60)
    print(
        f"WebSocket: ws://localhost:{PORT}/ws"
    )
    print("=" * 60)
    print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )
