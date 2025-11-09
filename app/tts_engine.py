from pathlib import Path
import tempfile
import subprocess


def _tts_with_pyttsx3(text: str, out_path: str) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return True
    except Exception as e:
    
        print(f"[TTS] pyttsx3 failed, falling back to espeak-ng CLI: {e}")
        return False


def _tts_with_espeak_cli(text: str, out_path: str) -> None:
    subprocess.run(
        ["espeak-ng", "-v", "en", "-w", out_path, text],
        check=True
    )


def synth_to_wav(text: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name


    if not _tts_with_pyttsx3(text, out_path):
        _tts_with_espeak_cli(text, out_path)


    return Path(out_path)




