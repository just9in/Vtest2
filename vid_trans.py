import os
import subprocess
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS
import soundfile as sf
import numpy as np

# ================= CONFIG =================
VIDEO_FILE = "input.mp4"
WORK_DIR = "work"
os.makedirs(WORK_DIR, exist_ok=True)

# ==========================================
# 1. Extract audio
print("Extracting audio...")
audio_path = f"{WORK_DIR}/audio.wav"
subprocess.run([
    "ffmpeg", "-y", "-i", VIDEO_FILE,
    "-ar", "16000", "-ac", "1", audio_path
])

# ==========================================
# 2. Separate vocals using Demucs
print("Separating vocals...")
subprocess.run([
    "demucs", "--two-stems=vocals", audio_path, "-o", WORK_DIR
])

# Paths after demucs
base = os.path.splitext(os.path.basename(audio_path))[0]
vocals_path = f"{WORK_DIR}/htdemucs/{base}/vocals.wav"
bg_path     = f"{WORK_DIR}/htdemucs/{base}/no_vocals.wav"

# ==========================================
# 3. Transcribe with timestamps
print("Transcribing...")
whisper = WhisperModel("medium", compute_type="float16")

segments, _ = whisper.transcribe(vocals_path)

# ==========================================
# 4. Load translation model (NLLB)
print("Loading translation model...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
translator = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = translator.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["ben_Beng"]
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

# ==========================================
# 5. Load TTS (Bengali supported model)
print("Loading TTS...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# ==========================================
# 6. Generate speech per segment with timing
print("Generating Bengali speech...")

final_audio = np.zeros(16000 * 60 * 30)  # supports up to 30 min

for seg in segments:
    start = int(seg.start * 16000)
    end   = int(seg.end * 16000)

    eng_text = seg.text.strip()
    if not eng_text:
        continue

    bn_text = translate(eng_text)

    wav = tts.tts(bn_text)
    wav = np.array(wav)

    # fit into segment duration
    seg_len = end - start
    if len(wav) > seg_len:
        wav = wav[:seg_len]
    else:
        pad = np.zeros(seg_len - len(wav))
        wav = np.concatenate([wav, pad])

    final_audio[start:start+len(wav)] += wav

# save speech track
speech_path = f"{WORK_DIR}/bengali_speech.wav"
sf.write(speech_path, final_audio, 16000)

# ==========================================
# 7. Mix speech + background
print("Mixing audio...")

mixed_path = f"{WORK_DIR}/final_audio.wav"
subprocess.run([
    "ffmpeg", "-y",
    "-i", speech_path,
    "-i", bg_path,
    "-filter_complex", "amix=inputs=2:duration=longest",
    mixed_path
])

# ==========================================
# 8. Attach back to video
print("Merging with video...")

output_video = "output_bengali.mp4"
subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_FILE,
    "-i", mixed_path,
    "-c:v", "copy",
    "-map", "0:v:0",
    "-map", "1:a:0",
    output_video
])

print("DONE:", output_video)