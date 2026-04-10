# test_piper_windows.py
import io, wave, sounddevice as sd, numpy as np
from piper import PiperVoice

voice = PiperVoice.load("./voices/en_US-lessac-medium.onnx")

buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(22050)
    for chunk in voice.synthesize("Hello, Piper is working on Windows."):
        wf.writeframes(chunk.audio_int16_bytes)

buf.seek(0)
data = buf.read()
print(f"WAV bytes: {len(data)}")   # should be >> 44

# also try playing it directly
buf.seek(0)
with wave.open(buf, "rb") as wf:
    rate = wf.getframerate()
    frames = wf.readframes(wf.getnframes())

audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
print("Playing...")
sd.play(audio, samplerate=rate)
sd.wait()
print("Done.")