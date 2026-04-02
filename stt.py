import sounddevice as sd
import torch
import numpy as np
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel
import threading

# =========================
# CONFIG (tune if needed)
# =========================
SAMPLING_RATE = 16000
CHUNK_SIZE = 512  # ~32ms (low latency)

THRESHOLD = 0.5
MIN_SILENCE_MS = 400
SPEECH_PAD_MS = 100

# =========================
# LOAD MODEL
# =========================
model = load_silero_vad()

vad = VADIterator(
    model,
    threshold=THRESHOLD,
    min_silence_duration_ms=MIN_SILENCE_MS,
    speech_pad_ms=SPEECH_PAD_MS
)

model = WhisperModel(
    "medium",              # or "small", "large-v3"
    device="cpu",         # "cpu" if no GPU
    compute_type="int8" # "int8" for low VRAM
)

# =========================
# STATE
# =========================
is_recording = False
audio_buffer = []

# =========================
# YOUR PROCESS FUNCTION
# =========================
# def process_audio(audio_np):
#     print(f"\n🧠 Processing {len(audio_np)/SAMPLING_RATE:.2f}s audio")

#     # 👉 plug Whisper here
#     # example:
#     # result = whisper_model.transcribe(audio_np)

#     # placeholder
#     print("🚀 Send this to Whisper / LLM\n")


def process_audio(audio_np):
    print(f"\n🧠 Transcribing {len(audio_np)/16000:.2f}s audio")

    segments, info = model.transcribe(
        audio_np,
        beam_size=5,
        vad_filter=False  # we already using Silero
    )

    full_text = ""

    for segment in segments:
        full_text += segment.text + " "

    print(f"📝 {full_text.strip()}\n")

    # 👉 next step: send to LLM API



# =========================
# AUDIO CALLBACK
# =========================
def audio_callback(indata, frames, time, status):
    global is_recording, audio_buffer

    if status:
        print(status)

    # get mono channel
    audio = indata[:, 0]

    # convert → torch
    audio_torch = torch.from_numpy(audio).float()

    # run VAD
    speech_dict = vad(audio_torch, return_seconds=True)

    # -------------------------
    # START DETECTED
    # -------------------------
    if speech_dict and "start" in speech_dict:
        print(f"🔥 Speech START")

        is_recording = True
        audio_buffer = []

    # -------------------------
    # RECORD AUDIO
    # -------------------------
    if is_recording:
        audio_buffer.extend(audio)

    # -------------------------
    # END DETECTED
    # -------------------------
    if speech_dict and "end" in speech_dict:
        print(f"🛑 Speech END")

        is_recording = False

        if len(audio_buffer) > 0:
            audio_np = np.array(audio_buffer, dtype=np.float32)

            # process async (recommended)
            process_audio(audio_np)


# =========================
# START STREAM
# =========================
with sd.InputStream(
    samplerate=SAMPLING_RATE,
    blocksize=CHUNK_SIZE,
    channels=1,
    dtype='float32',
    callback=audio_callback
):
    print("🎤 Listening...")

    while True:
        pass