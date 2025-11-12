import uvicorn
from fastapi import FastAPI, File, UploadFile, status, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.kokoro_tts import TTS

from app.asr_distilwhisper import transcribe_bytes
from app.tts_engine import synth_to_wav

app = FastAPI(title="MS1 – Distil-Whisper + pyttsx3 (FastAPI)")

# Serve /static (JS/CSS/etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    # Serve the HTML file without decoding it in Python
    return FileResponse("static/index.html", media_type="text/html; charset=utf-8")

@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    ext = audio.filename.split(".")[-1] if "." in audio.filename else "webm"
    data = await audio.read()
    text = transcribe_bytes(data, input_ext=ext)
    print(f"[ASR] {text}")
    return {"text": text}

@app.post("/tts")
async def tts(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    wav = synth_to_wav(text)
    return FileResponse(wav, media_type="audio/wav", filename="reply.wav")

@app.post("/ktts")
async def speak(payload: dict, response: Response):
    text = (payload.get("text") or "").strip()
    print("speaking")
    result = ktts.speak(text)
    print("Passed here")
    response.status_code = result.get("status", status.HTTP_400_BAD_REQUEST)

@app.post("/ask")
async def ask(audio: UploadFile = File(...)):
    ext = audio.filename.split(".")[-1] if "." in audio.filename else "webm"
    data = await audio.read()
    text = transcribe_bytes(data, input_ext=ext)
    print(f"[ASR] {text}")
    return {"text": text}


if __name__ == '__main__':
    ktts = TTS()
    uvicorn.run(app, port=8000)