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

# Silero VAD chunk.
VAD_CHUNK_SAMPLES = 512       # 32 ms @ 16 kHz

# Silence required before speech ends.
MIN_SILENCE_MS = 600

# Extra audio around speech boundaries.
SPEECH_PAD_MS = 200


# ============================================================
# Rolling interim configuration
# ============================================================

# Minimum time between interim requests.
INTERIM_INTERVAL = 0.8

# Only the newest portion of the utterance is used for
# interim recognition.
INTERIM_WINDOW_SECONDS = 2.5

# Don't try to recognize extremely short audio.
INTERIM_MIN_AUDIO_SECONDS = 0.8


# ============================================================
# Whisper
# ============================================================

WHISPER_MODEL = "models/ggml-base.bin"

# Language code like "en" or "pl"
# Assumes all speech is in this language.
# Set to None to enable automatic language detection.
WHISPER_LANGUAGE = "en"


# ============================================================
# Application state
# ============================================================

app = FastAPI()

audio_queue = queue.Queue()

clients = set()

event_loop = None


# ============================================================
# Whisper job system
# ============================================================

# A request for an interim transcription.
#
# This is deliberately NOT a normal queue.
#
# There should never be a backlog of old interim jobs.
interim_lock = threading.Lock()

interim_pending = None

interim_running = False


# Final transcription has its own queue.
#
# There should only ever be one final job waiting.
final_queue = queue.Queue(maxsize=1)


# Only one Whisper call may happen at a time.
whisper_lock = threading.Lock()


# Used to invalidate old interim jobs.
speech_generation = 0

generation_lock = threading.Lock()


# ============================================================
# Load models
# ============================================================

print(
    "Loading Whisper.cpp Vulkan GPU model...",
    flush=True,
)

whisper = WhisperGPU(WHISPER_MODEL)

print(
    "Whisper.cpp GPU model loaded.",
    flush=True,
)


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

def microphone_callback(
    indata,
    frames,
    time_info,
    status,
):
    if status:
        print(
            f"\n[AUDIO] {status}",
            flush=True,
        )

    # sounddevice reuses its input buffer,
    # so make a copy.
    audio = indata[:, 0].copy()

    audio_queue.put(audio)


def microphone_worker():
    print(
        "Starting microphone...",
        flush=True,
    )

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=VAD_CHUNK_SAMPLES,
            callback=microphone_callback,
        ):
            print(
                "Microphone running.",
                flush=True,
            )

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
    Run whisper.cpp on a float32 mono numpy array.

    This function is thread-safe because all Whisper calls
    are serialized through whisper_lock.
    """

    if len(audio) == 0:
        return ""

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    with whisper_lock:
        return whisper.transcribe(audio, language=WHISPER_LANGUAGE)


# ============================================================
# Generation handling
# ============================================================

def get_generation():
    with generation_lock:
        return speech_generation


def new_generation():
    global speech_generation

    with generation_lock:
        speech_generation += 1
        return speech_generation


# ============================================================
# Interim scheduling
# ============================================================

def schedule_interim(
    audio,
    speech_start_time,
    generation,
):
    """
    Replace any existing pending interim.

    There is intentionally only ONE pending interim.
    """

    global interim_pending

    with interim_lock:

        # If this speech session is already obsolete,
        # don't schedule anything.
        if generation != get_generation():
            return

        interim_pending = (
            audio,
            speech_start_time,
            generation,
        )


def cancel_pending_interim():
    """
    Remove a pending interim.

    This does NOT attempt to kill an inference that is
    already executing. Instead, generation invalidates
    its result.
    """

    global interim_pending

    with interim_lock:
        interim_pending = None


# ============================================================
# Interim worker
# ============================================================

def interim_worker():
    """
    Latest-only interim worker.

    If Whisper is busy, newer interim requests replace
    older pending requests rather than forming a backlog.
    """

    global interim_pending
    global interim_running

    print(
        "Interim Whisper worker started.",
        flush=True,
    )

    while True:

        job = None

        with interim_lock:

            if (
                interim_pending is not None
                and not interim_running
            ):
                job = interim_pending
                interim_pending = None
                interim_running = True

        if job is None:
            time.sleep(0.01)
            continue

        audio, speech_start_time, generation = job

        try:

            # Check again before doing expensive work.
            if generation != get_generation():
                continue

            (text, lang) = transcribe(audio)

            # The speech may have ended while Whisper
            # was processing this interim.
            if generation != get_generation():
                continue

            if not text:
                continue

            duration = len(audio) / SAMPLE_RATE

            # This is the beginning of the rolling window.
            start_offset = max(
                0.0,
                duration - INTERIM_WINDOW_SECONDS,
            )

            message = make_message(
                "interim",
                lang,
                speech_start_time + start_offset,
                speech_start_time + duration,
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

        finally:

            with interim_lock:
                interim_running = False


# ============================================================
# Final worker
# ============================================================

def final_worker():
    """
    Final transcription worker.

    Final jobs have priority because the VAD worker never
    performs Whisper inference itself.
    """

    print(
        "Final Whisper worker started.",
        flush=True,
    )

    while True:

        job = final_queue.get()

        if job is None:
            continue

        audio, speech_start_time, speech_end_time = job

        try:

            print(
                "[WHISPER] Running FINAL transcription...",
                flush=True,
            )

            (text, lang) = transcribe(audio)

            if not text:
                continue

            message = make_message(
                "final",
                lang,
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

        finally:

            final_queue.task_done()


# ============================================================
# Speech processing / VAD
# ============================================================

def speech_worker():
    print(
        "Speech/VAD worker started.",
        flush=True,
    )

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

    last_interim_request = 0.0

    generation = 0

    while True:

        audio = audio_queue.get()

        try:

            tensor = np.asarray(
                audio,
                dtype=np.float32,
            )

            import torch

            tensor_torch = torch.from_numpy(
                tensor
            )

            vad_event = vad(
                tensor_torch,
                return_seconds=False,
            )

            now = time.monotonic()

            # =================================================
            # Speech start
            # =================================================

            if (
                vad_event is not None
                and "start" in vad_event
            ):

                speech_active = True

                speech_start_time = (
                    vad_event["start"]
                    / SAMPLE_RATE
                )

                generation = new_generation()

                print(
                    f"\n[VAD] SPEECH START "
                    f"{speech_start_time:.2f}s",
                    flush=True,
                )

                speech_audio = [
                    tensor.copy()
                ]

                last_interim_request = now

                continue

            # =================================================
            # Collect audio
            # =================================================

            if speech_active:

                speech_audio.append(
                    tensor.copy()
                )

            # =================================================
            # Request an interim
            # =================================================

            if speech_active:

                elapsed = (
                    now - last_interim_request
                )

                total_samples = sum(
                    len(chunk)
                    for chunk in speech_audio
                )

                total_seconds = (
                    total_samples
                    / SAMPLE_RATE
                )

                if (
                    elapsed >= INTERIM_INTERVAL
                    and total_seconds
                    >= INTERIM_MIN_AUDIO_SECONDS
                ):

                    # -----------------------------------------
                    # Build rolling window.
                    # -----------------------------------------

                    window_samples = int(
                        INTERIM_WINDOW_SECONDS
                        * SAMPLE_RATE
                    )

                    start_sample = max(
                        0,
                        total_samples
                        - window_samples,
                    )

                    chunks = []

                    position = 0

                    for chunk in speech_audio:

                        chunk_end = (
                            position
                            + len(chunk)
                        )

                        if chunk_end > start_sample:

                            local_start = max(
                                0,
                                start_sample
                                - position,
                            )

                            chunks.append(
                                chunk[local_start:]
                            )

                        position = chunk_end

                    if chunks:

                        interim_audio = np.concatenate(
                            chunks
                        )

                        schedule_interim(
                            interim_audio,
                            speech_start_time,
                            generation,
                        )

                    last_interim_request = now

            # =================================================
            # Speech end
            # =================================================

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

                # ---------------------------------------------
                # Immediately invalidate every interim from
                # this speech session.
                # ---------------------------------------------

                new_generation()

                cancel_pending_interim()

                # ---------------------------------------------
                # Complete utterance.
                # ---------------------------------------------

                audio_data = np.concatenate(
                    speech_audio
                )

                # ---------------------------------------------
                # Reset state BEFORE queueing final.
                # ---------------------------------------------

                speech_active = False

                speech_audio = []

                speech_start = (
                    speech_start_time
                )

                speech_start_time = None

                last_interim_request = 0.0

                vad.reset_states()

                # ---------------------------------------------
                # Final gets priority.
                #
                # The queue should never contain multiple
                # final jobs.
                # ---------------------------------------------

                try:

                    # Remove an obsolete final if one somehow
                    # remains.
                    while True:
                        try:
                            final_queue.get_nowait()
                            final_queue.task_done()

                        except queue.Empty:
                            break

                    final_queue.put_nowait(
                        (
                            audio_data,
                            speech_start,
                            end_time,
                        )
                    )

                except queue.Full:

                    print(
                        "[WHISPER] Final queue full!",
                        flush=True,
                    )

        except Exception as e:

            print(
                f"\n[SPEECH ERROR] {e!r}",
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

    # Microphone.
    threading.Thread(
        target=microphone_worker,
        daemon=True,
    ).start()

    # VAD / audio processing.
    threading.Thread(
        target=speech_worker,
        daemon=True,
    ).start()

    # Latest-only interim Whisper worker.
    threading.Thread(
        target=interim_worker,
        daemon=True,
    ).start()

    # Final Whisper worker.
    threading.Thread(
        target=final_worker,
        daemon=True,
    ).start()

    print()

    print("=" * 60)

    print(
        f"WebSocket: ws://localhost:{PORT}/ws"
    )

    print(
        "Whisper backend: whisper.cpp / Vulkan"
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
