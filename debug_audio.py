import io, wave, requests
import numpy as np
import sounddevice as sd

BACKEND_URL = "http://100.80.81.61:8000/chat"  # your actual IP

# 1. Fetch audio from backend
resp = requests.post(BACKEND_URL, json={"text": "say hello"}, timeout=60)
print(f"Status: {resp.status_code}, bytes received: {len(resp.content)}")

# 2. Save to disk so we can inspect it
with open("test_output.wav", "wb") as f:
    f.write(resp.content)
print("Saved to test_output.wav — try opening it manually")

# 3. Decode and print WAV properties
buf = io.BytesIO(resp.content)
with wave.open(buf, "rb") as wf:
    rate     = wf.getframerate()
    channels = wf.getnchannels()
    width    = wf.getsampwidth()
    n_frames = wf.getnframes()
    frames   = wf.readframes(n_frames)
    print(f"WAV: rate={rate}, channels={channels}, width={width}, frames={n_frames}")

# 4. Attempt playback
audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
if channels > 1:
    audio = audio.reshape(-1, channels)
print(f"Audio array shape: {audio.shape}")
print("Playing...")
sd.play(audio, samplerate=rate)
sd.wait()
print("Done.")