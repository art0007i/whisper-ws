# server.py

import asyncio
import json
import queue
import threading

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel


HOST = "0.0.0.0"
PORT = 6912

SAMPLE_RATE = 16000
CHANNELS = 1

# Audio chunks given to Whisper.
CHUNK_SECONDS = 3

MODEL_NAME = "small"


app = FastAPI()

audio_queue = queue.Queue()
clients = set()

loop = None


print(f"Loading Whisper model: {MODEL_NAME}...", flush=True)

model = WhisperModel(
    MODEL_NAME,
    device="cpu",
    compute_type="int8",
)

print("Whisper model loaded.", flush=True)


# ------------------------------------------------------------
# Microphone
# ------------------------------------------------------------

def microphone_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio: {status}", flush=True)

    audio = indata[:, 0].copy()

    audio_queue.put(audio)


def microphone_thread():
    print("Starting microphone...", flush=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * CHUNK_SECONDS),
            callback=microphone_callback,
        ):
            print("Microphone is running.", flush=True)

            while True:
                sd.sleep(1000)

    except Exception as e:
        print(f"MICROPHONE ERROR: {e!r}", flush=True)


# ------------------------------------------------------------
# Whisper
# ------------------------------------------------------------

def whisper_thread():
    print("Whisper worker started.", flush=True)

    while True:
        audio = audio_queue.get()

        try:
            segments, info = model.transcribe(
                audio,
                beam_size=1,
                vad_filter=True,
            )

            for segment in segments:
                text = segment.text.strip()

                if not text:
                    continue

                message = """lang=%s
start=%.2f
end=%.2f
text=%s
""" % (info.language, segment.start, segment.end, text)

                print(f">>> {text}", flush=True)

                if loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast(message),
                        loop,
                    )

        except Exception as e:
            print(f"WHISPER ERROR: {e!r}", flush=True)


# ------------------------------------------------------------
# WebSocket
# ------------------------------------------------------------

async def broadcast(message):
    if not clients:
        return

    disconnected = []

    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        clients.discard(client)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    clients.add(websocket)

    print(
        f"WebSocket connected. Clients: {len(clients)}",
        flush=True,
    )

    try:
        while True:
            # We don't actually need messages from the client.
            # Receiving keeps the connection alive and lets us
            # detect disconnects.
            await websocket.receive()

    except WebSocketDisconnect:
        pass

    except Exception as e:
        print(f"WEBSOCKET ERROR: {e!r}", flush=True)

    finally:
        clients.discard(websocket)

        print(
            f"WebSocket disconnected. Clients: {len(clients)}",
            flush=True,
        )


# ------------------------------------------------------------
# Startup
# ------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global loop

    loop = asyncio.get_running_loop()

    threading.Thread(
        target=microphone_thread,
        daemon=True,
    ).start()

    threading.Thread(
        target=whisper_thread,
        daemon=True,
    ).start()

    print("", flush=True)
    print("=" * 50, flush=True)
    print(f"WebSocket server: ws://localhost:{PORT}/ws", flush=True)
    print("=" * 50, flush=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )
