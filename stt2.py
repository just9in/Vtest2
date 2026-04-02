#vad + whisper
import sounddevice as sd
import torch
import numpy as np
import threading
import time
import queue
import os
import warnings
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel

# =========================
# CONFIG
# =========================
SAMPLING_RATE = 16000
# Silero VAD expects exactly 512 samples per frame at 16kHz (or 256 at 8kHz).
CHUNK_SIZE = 512  # 32ms

# Set True if you want to see VAD/stream debug logs.
DEBUG = False

# Reduce third-party console noise (model downloads/caching warnings).
if not DEBUG:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    warnings.filterwarnings("ignore", module="huggingface_hub")

THRESHOLD = 0.5
MIN_SILENCE_MS = 400
SPEECH_PAD_MS = 100

# Optional post-VAD gating to avoid Whisper hallucinations on very quiet/noisy audio.
MIN_UTTERANCE_SECONDS = 0.25
MIN_RMS = 0.002
NO_SPEECH_PROB_THRESHOLD = 0.60

WHISPER_MODEL_SIZE = "small"   # "small", "medium", "large-v3"
DEVICE = "cpu"                 # "cpu" if no GPU
COMPUTE_TYPE = "int8"        # "int8" for low VRAM

# =========================
# LOAD MODELS (ONLY ONCE)
# =========================
print("Loading models...")

vad_model = load_silero_vad()

vad = VADIterator(
    vad_model,
    threshold=THRESHOLD,
    min_silence_duration_ms=MIN_SILENCE_MS,
    speech_pad_ms=SPEECH_PAD_MS
)

whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print("Ready.")

# =========================
# STATE
# =========================
is_recording = False
audio_buffer = []
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)
_dropped_utterances = 0

# =========================
# WHISPER FUNCTION
# =========================
def process_audio(audio_np):
    duration_s = len(audio_np) / SAMPLING_RATE
    print(f'duration_s={duration_s}')
    if duration_s < MIN_UTTERANCE_SECONDS:
        if DEBUG:
            print(f"(skipping short audio: {duration_s:.2f}s)", flush=True)
        return

    rms = float(np.sqrt(np.mean(np.square(audio_np))))
    if rms < MIN_RMS:
        if DEBUG:
            print(f"(skipping quiet audio: rms={rms:.4f})", flush=True)
        return

    if DEBUG:
        print(f"\n🧠 Transcribing {len(audio_np)/SAMPLING_RATE:.2f}s audio", flush=True)

    segments, info = whisper_model.transcribe(
        audio_np,
        language="en",
        beam_size=5,
        vad_filter=False
    )

    parts = []
    for segment in segments:
        # Filter out hallucinated segments on (near) silence.
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        if no_speech_prob is not None and no_speech_prob >= NO_SPEECH_PROB_THRESHOLD:
            continue
        if segment.text:
            parts.append(segment.text.strip())

    text = " ".join(p for p in parts if p).strip()
    if text:
        print(text, flush=True)
    elif DEBUG:
        print("(no speech detected)", flush=True)

    # 👉 NEXT: send this text to your LLM API

def transcription_worker():
    while True:
        item = audio_queue.get()
        if item is None:
            return
        try:
            process_audio(item)
        finally:
            audio_queue.task_done()


threading.Thread(target=transcription_worker, daemon=True).start()

# =========================
# AUDIO CALLBACK
# =========================
def audio_callback(indata, frames, time, status):
    global is_recording, audio_buffer, _dropped_utterances

    # `status` can contain "input overflow" when the app can't keep up.
    # For clean output (and to avoid spam), we suppress it by default.
    if DEBUG and status:
        print(status, flush=True)

    audio = indata[:, 0]
    audio_torch = torch.from_numpy(audio)

    speech_dict = vad(audio_torch, return_seconds=True)

    # START
    if speech_dict and "start" in speech_dict:
        if DEBUG:
            print("🔥 Speech START", flush=True)
        is_recording = True
        audio_buffer = []

    # RECORD
    if is_recording:
        audio_buffer.extend(audio)

    # END
    if speech_dict and "end" in speech_dict:
        if DEBUG:
            print("🛑 Speech END", flush=True)
        is_recording = False

        if len(audio_buffer) > 0:
            audio_np = np.array(audio_buffer, dtype=np.float32)

            # Queue for transcription (single worker thread).
            try:
                audio_queue.put_nowait(audio_np)
            except queue.Full:
                _dropped_utterances += 1
                if DEBUG:
                    print(f"⚠️ Dropping utterance (queue full). Dropped: {_dropped_utterances}", flush=True)

# =========================
# START STREAM
# =========================
with sd.InputStream(
    samplerate=SAMPLING_RATE,
    blocksize=CHUNK_SIZE,
    channels=1,
    dtype='float32',
    latency='high',
    callback=audio_callback
):
    print("Listening... (press Ctrl+C to stop)")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass