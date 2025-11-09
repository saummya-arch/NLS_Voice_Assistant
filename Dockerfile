FROM python:3.11-slim


# System deps for ASR + TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng espeak-ng-data libespeak-ng1 \
    libespeak1 \
  && rm -rf /var/lib/apt/lists/*


# (Optional) If pyttsx3 still looks for libespeak.so.1 only, ensure a symlink exists
RUN if [ ! -e /usr/lib/x86_64-linux-gnu/libespeak.so.1 ] && [ -e /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 ]; then \
      ln -s /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 /usr/lib/x86_64-linux-gnu/libespeak.so.1; \
    fi


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY app/ app/


# Distil-Whisper model (change if you like)
ENV DISTIL_MODEL=distil-whisper/distil-medium.en
ENV PYTHONUNBUFFERED=1


# Pre-cache model so it runs offline
RUN python - << 'PY'
import os
from transformers import pipeline
model = os.getenv("DISTIL_MODEL", "distil-whisper/distil-medium.en")
print("Pre-caching:", model)
_ = pipeline("automatic-speech-recognition", model=model, device=-1)
print("Cached.")
PY


EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


