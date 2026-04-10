import io
import re
import json
import queue
import struct
import wave
import numpy as np
import requests
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from piper import PiperVoice

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000
CHUNK_SIZE    = 512

WHISPER_MODEL   = "base.en"
WHISPER_DEVICE  = "cpu"
WHISPER_COMPUTE = "int8"

VAD_THRESHOLD      = 0.5
VAD_MIN_SILENCE_MS = 600

OLLAMA_URL   = "https://ollama.ubuntudevt65535.dpdns.org/api/generate"
OLLAMA_MODEL = "llama3:latest"

VOICE_MODEL  = "./voices/en_US-lessac-medium.onnx"
VOICE_RATE   = 22050   # lessac-medium outputs 22050 Hz

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# ─── Shared state ─────────────────────────────────────────────────────────────
audio_queue      = queue.Queue()
speech_frames    = []
is_speaking      = False
is_playing       = False
conversation     = []   # full history for multi-turn context

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

print("Loading Piper TTS...")
voice = PiperVoice.load(VOICE_MODEL)

print("Ready. Speak into your mic. Ctrl+C to stop.\n")

# ─── Mic callback ─────────────────────────────────────────────────────────────
def mic_callback(indata, frames, time_info, status):
    if status:
        print(f"[MIC] {status}")
    if not is_playing:
        audio_queue.put(indata[:, 0].copy())

# ─── TTS — synthesize one sentence to a numpy array ───────────────────────────
def synthesize(text: str) -> np.ndarray | None:
    pcm = bytearray()
    for chunk in voice.synthesize(text):
        pcm.extend(chunk.audio_int16_bytes)
    if not pcm:
        return None
    return np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0

# ─── LLM — stream tokens, synthesize + play each sentence as it arrives ───────
def stream_and_speak(user_text: str) -> None:
    global is_playing, conversation

    # Build prompt from conversation history
    conversation.append({"role": "user", "content": user_text})
    prompt = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in conversation
    ) + "\nAssistant:"

    full_reply   = ""
    sentence_buf = ""
    is_playing   = True

    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue

                data         = json.loads(line)
                token        = data.get("response", "")
                done         = data.get("done", False)
                full_reply   += token
                sentence_buf += token

                # Split on sentence boundaries and speak each complete sentence
                parts = SENTENCE_SPLIT.split(sentence_buf)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        print(f"[TTS] {sentence}")
                        audio = synthesize(sentence)
                        if audio is not None:
                            sd.play(audio, samplerate=VOICE_RATE)
                            sd.wait()
                    sentence_buf = parts[-1]

                if done:
                    break

        # Speak any remaining text after stream ends
        sentence_buf = sentence_buf.strip()
        if sentence_buf:
            print(f"[TTS] {sentence_buf}")
            audio = synthesize(sentence_buf)
            if audio is not None:
                sd.play(audio, samplerate=VOICE_RATE)
                sd.wait()

        conversation.append({"role": "assistant", "content": full_reply.strip()})
        print(f"[DONE] {full_reply.strip()}\n")

    except requests.RequestException as e:
        print(f"[ERROR] LLM request failed: {e}")

    finally:
        is_playing = False

# ─── ASR ──────────────────────────────────────────────────────────────────────
def transcribe_and_respond(frames: list[np.ndarray]) -> None:
    audio = np.concatenate(frames)
    segments, _ = whisper.transcribe(audio, language="en", beam_size=1)
    text = " ".join(seg.text.strip() for seg in segments)

    if not text:
        return

    print(f"[ASR] {text}")
    stream_and_speak(text)

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
                transcribe_and_respond(speech_frames)
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