import asyncio
import json
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator
import uvicorn


# ============================================================
# Configuration
# ============================================================

HOST = "0.0.0.0"
PORT = 6912

SAMPLE_RATE = 16000
CHANNELS = 1

# Silero VAD wants small chunks.
VAD_CHUNK_SAMPLES = 512       # 32 ms at 16 kHz

# Silence required before VAD considers speech finished.
MIN_SILENCE_MS = 600

# Extra audio around speech boundaries.
SPEECH_PAD_MS = 200

WHISPER_MODEL = "small"


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

print("Loading Whisper...", flush=True)

# Slower cpu model - but more compatible

whisper = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8",
)
"""
# Faster gpu model - but it might explode especially on amd
whisper = WhisperModel(
    WHISPER_MODEL,
    device="cuda",
    compute_type="float16",
)
"""
print("Whisper loaded.", flush=True)


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
    # IMPORTANT:
    # text is deliberately the final field.
    #
    # This means text can contain arbitrary '=' or newlines.
    # The receiver can simply find "text=" and take everything
    # after it.

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

    # Copy because sounddevice reuses its buffer.
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
        print(f"\n[MICROPHONE ERROR] {e!r}", flush=True)


# ============================================================
# Whisper
# ============================================================

def transcribe(audio):
    """
    Run Whisper on a numpy float32 mono array.
    """

    with whisper_lock:
        segments, info = whisper.transcribe(
            audio,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        segments = list(segments)

    text_parts = []

    start = None
    end = None

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        text_parts.append(text)

        if start is None:
            start = segment.start

        end = segment.end

    text = " ".join(text_parts).strip()

    return text, info.language, start, end


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

    while True:
        audio = audio_queue.get()

        try:
            tensor = np.asarray(audio, dtype=np.float32)

            import torch

            tensor = torch.from_numpy(tensor)

            vad_event = vad(tensor, return_seconds=False)

            # ------------------------------------------------
            # VAD detected speech start
            # ------------------------------------------------

            if vad_event is not None and "start" in vad_event:
                speech_active = True

                # VADIterator gives sample position.
                speech_start_time = (
                    vad_event["start"] / SAMPLE_RATE
                )

                print(
                    f"\n[VAD] SPEECH START "
                    f"{speech_start_time:.2f}s",
                    flush=True,
                )

                speech_audio = []

                # Start collecting from this point.
                speech_audio.append(tensor.numpy())

                continue

            if speech_active:
                speech_audio.append(tensor.numpy())

            # ------------------------------------------------
            # VAD detected speech end
            # ------------------------------------------------

            if vad_event is not None and "end" in vad_event:
                if not speech_active:
                    continue

                end_time = (
                    vad_event["end"] / SAMPLE_RATE
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

                # Reset VAD state for next utterance.
                vad.reset_states()

        except Exception as e:
            print(
                f"\n[SPEECH ERROR] {e!r}",
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
        text, language, start, end = transcribe(audio)

        if not text:
            return

        if start is None:
            start = 0

        if end is None:
            end = len(audio) / SAMPLE_RATE

        start += speech_start_time
        end += speech_start_time

        message = make_message(
            "final",
            language,
            start,
            end,
            text,
        )

        print(
            f"[FINAL] {text}",
            flush=True,
        )

        broadcast_from_thread(message)

    except Exception as e:
        print(
            f"[FINAL ERROR] {e!r}",
            flush=True,
        )


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    clients.add(websocket)

    print(
        f"\n[WS] Client connected "
        f"({len(clients)} clients)",
        flush=True,
    )

    try:
        while True:
            # Clients don't send audio.
            # We only need to detect disconnects.
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
