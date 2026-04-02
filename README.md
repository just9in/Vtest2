# Voice STT → Backend Chat → TTS Playback (Windows)

This repo is a simple voice pipeline:

1. Records microphone audio in real time.
2. Uses Silero VAD (voice activity detection) to find speech segments.
3. Transcribes speech locally with `faster-whisper`.
4. Sends the recognized text to a backend HTTP endpoint (`/chat`).
5. Plays back the WAV audio bytes returned by the backend.

It’s designed to avoid feedback: while backend audio is playing, mic frames are ignored.

## Project files

- `stt_pipeline.py` — MAIN FILE “listen → transcribe → POST text → play WAV” pipeline. 
- `stt2.py` — a simpler VAD + Whisper script (prints transcription only).
- `stt.py` — an older/alternate VAD + Whisper script (prints transcription only).
- `debug_audio.py` — calls the backend and verifies the returned WAV can be saved/played.
- `voices/` — ONNX voice model files (not currently used by the Python scripts in this repo; typically used by a TTS backend).

## Prerequisites

- Windows 10/11
- Python 3.10+ recommended
- A working microphone
- Network access to your backend server (often via Tailscale)

## Setup (venv)

From PowerShell in the repo folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Core deps used by stt_pipeline.py
pip install sounddevice numpy requests
pip install faster-whisper torch

# If you run stt.py or stt2.py (they import silero_vad)
pip install silero-vad
```

Notes:
- The first run will download Silero VAD via `torch.hub`.
- Installing `torch` on Windows can be the slowest step. If you prefer a CPU-only wheel, use the official PyTorch instructions for your Python version.

## Configure the backend URL

Update `BACKEND_URL` in:

- `stt_pipeline.py`
- `debug_audio.py`

Example:

```py
BACKEND_URL = "http://100.x.x.x:8000/chat"
```

The backend is expected to:

- Accept `POST /chat` with JSON like `{ "text": "hello" }`
- Return a WAV file as raw bytes in the response body

## Run

### Full voice pipeline

```powershell
python stt_pipeline.py
```

Expected console flow:

- “Loading Silero VAD…”
- “Loading Whisper…”
- Then it prints VAD start/end events, the transcription (`[ASR] ...`), and plays the backend’s audio.

Stop with `Ctrl+C`.

### Backend audio sanity check

```powershell
python debug_audio.py
```

This writes `test_output.wav` and attempts playback.

## Troubleshooting

- No mic audio / device errors: close other apps using the mic and re-run.
- “Backend unreachable”: confirm the IP/port, and that you’re connected to the same network (e.g., Tailscale).
- Choppy audio / dropouts: try increasing `CHUNK_SIZE` in `stt_pipeline.py` (higher latency, more stability).

## Git / GitHub notes

- A `.gitignore` is included to avoid committing your virtual environment and caches.
- The files in `voices/` can be large. If you plan to version models in GitHub, consider using Git LFS, or leave them out by uncommenting the related lines in `.gitignore`.
