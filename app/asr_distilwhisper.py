import os
import tempfile
from pathlib import Path
import torch
from transformers import pipeline

MODEL_NAME = os.getenv("DISTIL_MODEL", "distil-whisper/distil-medium.en")


_device = 0 if torch.cuda.is_available() else -1
_asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    device=_device,
)

def transcribe_bytes(audio_bytes: bytes, input_ext: str = "webm") -> str:
    """
    Save uploaded audio to a CLOSED temp file that the pipeline can open (Windows-safe).
    """
    fd, path = tempfile.mkstemp(suffix=f".{input_ext}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)  # closes at the end of with-block
        # Transformers pipeline can take a path
        result = _asr(path)
        text = result.get("text") if isinstance(result, dict) else str(result)
        return (text or "").strip()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

