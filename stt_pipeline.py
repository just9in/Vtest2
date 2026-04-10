import io
import queue
import wave
import numpy as np
import requests
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000
CHUNK_SIZE    = 512

WHISPER_MODEL   = "base.en"
WHISPER_DEVICE  = "cpu"
WHISPER_COMPUTE = "int8"

VAD_THRESHOLD         = 0.5
VAD_MIN_SILENCE_MS    = 600

BACKEND_URL = "http://100.100.53.80:8000/chat"   # ← paste your Tailscale IP here

# ─── Shared state ─────────────────────────────────────────────────────────────
audio_queue   = queue.Queue()
speech_frames = []
is_speaking   = False
is_playing    = False    # mic ignores VAD while TTS is playing (prevents feedback)

# ─── Load models ──────────────────────────────────────────────────────────────
print("Loading Silero VAD...")
vad_model, utils = torch.hub.load(
    repo_or_dir  = "snakers4/silero-vad",
    model        = "silero_vad",
    force_reload = False,
)
VADIterator  = utils[3]
vad_iterator = VADIterator(
    model                   = vad_model,
    sampling_rate           = SAMPLE_RATE,
    threshold               = VAD_THRESHOLD,
    min_silence_duration_ms = VAD_MIN_SILENCE_MS,
)

print("Loading Whisper...")
whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print(f"Ready. Talking to backend at {BACKEND_URL}\n")

# ─── Mic callback ─────────────────────────────────────────────────────────────
def mic_callback(indata, frames, time_info, status):
    if status:
        print(f"[MIC] {status}")
    if not is_playing:                         # drop frames while speaker is active
        audio_queue.put(indata[:, 0].copy())

# ─── Play WAV bytes returned from the backend ─────────────────────────────────
def play_wav(wav_bytes: bytes) -> None:
    global is_playing
    is_playing = True

    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        rate   = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, samplerate=rate)
    sd.wait()                                  # blocks until playback finishes

    is_playing = False

# ─── Transcribe + send to backend ─────────────────────────────────────────────
def transcribe_and_chat(frames: list[np.ndarray]) -> None:
    audio = np.concatenate(frames)

    segments, _ = whisper.transcribe(audio, language="en", beam_size=1)
    text = " ".join(seg.text.strip() for seg in segments)

    if not text:
        return

    print(f"[ASR]     {text}")

    try:
        resp = requests.post(BACKEND_URL, json={"text": text}, timeout=60)
        resp.raise_for_status()
        print(f"[PLAYING] response audio...")
        play_wav(resp.content)
    except requests.RequestException as e:
        print(f"[ERROR] Backend unreachable: {e}")

# ─── VAD loop ─────────────────────────────────────────────────────────────────
def vad_loop() -> None:
    global is_speaking, speech_frames

    while True:
        chunk  = audio_queue.get()
        tensor = torch.from_numpy(chunk).float()

        speech_event = vad_iterator(tensor, return_seconds=False)

        if is_speaking:
            speech_frames.append(chunk)

        if speech_event:
            if "start" in speech_event:
                print("[VAD] Speech start")
                is_speaking   = True
                speech_frames = [chunk]

            if "end" in speech_event:
                print("[VAD] Speech end → transcribing...")
                is_speaking = False
                transcribe_and_chat(speech_frames)
                speech_frames = []
                vad_iterator.reset_states()

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "float32",
        blocksize  = CHUNK_SIZE,
        callback   = mic_callback,
    ):
        try:
            vad_loop()
        except KeyboardInterrupt:
            print("\nStopped.")
# ```

# ---

# ## What to expect when it runs
# ```
# Loading Silero VAD...
# Loading Whisper...
# Ready. Talking to backend at http://100.x.x.x:8000/chat

# [VAD] Speech start
# [VAD] Speech end → transcribing...
# [ASR]     What is the capital of France?
# [PLAYING] response audio...
# ```

# And the Linux terminal will show:
# ```
# [IN]  What is the capital of France?
# [OUT] The capital of France is Paris...