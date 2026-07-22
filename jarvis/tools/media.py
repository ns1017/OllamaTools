"""Media ingestion tools: audio format conversion + Vosk transcription,
image loading for vision-capable Ollama calls, and plain-text file reading."""
import base64
import json
import os
import shutil
import subprocess
import wave

from vosk import KaldiRecognizer

from jarvis.capabilities import FFMPEG_AVAILABLE, FFMPEG_PATH
from jarvis.config import CONFIG
from jarvis.logging_utils import log_event

TEXT_FILE_EXTENSIONS = set(CONFIG.get("text_file_extensions", default=[
    ".txt", ".md", ".csv", ".json", ".log", ".py", ".yaml", ".yml", ".ini", ".cfg"
]))


def _is_path_allowed(path: str) -> bool:
    """Confine file access to an explicit allowlist of directories
    (config.json -> security.file_access_allowed_roots). read_file,
    ingest_image, and transcribe_voice all take an arbitrary path straight
    from model output — without this check, a crafted '../../' or absolute
    path could walk the model outside its own working area and read
    anything the host OS user can see."""
    if not path:
        return False

    blocked_names = CONFIG.get("security", "file_access_blocked_names", default=[])
    basename = os.path.basename(path).lower()
    if basename in [b.lower() for b in blocked_names]:
        return False

    allowed_roots = CONFIG.get("security", "file_access_allowed_roots", default=["."])
    try:
        real_path = os.path.realpath(path)
    except Exception:
        return False

    for root in allowed_roots:
        try:
            real_root = os.path.realpath(root)
        except Exception:
            continue
        if real_path == real_root or real_path.startswith(real_root + os.sep):
            return True
    return False


def convert_to_wav_16k_mono(src_path: str):
    """Convert any audio file (Telegram voice notes are OGG/Opus) to the
    16kHz mono PCM16 WAV format Vosk requires. Needs ffmpeg on PATH.
    Note: ffmpeg -i detects the input container/codec by probing the file's
    contents, not its extension, so .oga/.ogg/.mp3/etc. all work the same way —
    no format-specific branching needed here."""
    ffmpeg_path = shutil.which("ffmpeg") or (FFMPEG_PATH if FFMPEG_AVAILABLE else None)
    if not ffmpeg_path:
        log_event("audio:ffmpeg_missing", {"src": src_path})
        print("❌ ffmpeg not found on PATH for this process.")
        print("   If you just added it to PATH, close and reopen your terminal/IDE")
        print("   (and restart this script) — already-open shells keep the old PATH.")
        print("   Verify with: ffmpeg -version   (run that in the same terminal you launch this script from)")
        return None
    if not os.path.exists(src_path):
        log_event("audio:src_missing", {"src": src_path})
        print(f"❌ Source audio file does not exist: {src_path}")
        return None
    dst_path = os.path.splitext(src_path)[0] + "_16k.wav"
    try:
        result = subprocess.run(
            [ffmpeg_path, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 or not os.path.exists(dst_path):
            log_event("audio:convert_failed", {"src": src_path, "stderr": result.stderr[-500:]})
            print(f"❌ ffmpeg conversion failed (return code {result.returncode}) for {src_path}")
            print("---- ffmpeg stderr (last 800 chars) ----")
            print(result.stderr.strip()[-800:])
            print("-----------------------------------------")
            return None
        return dst_path
    except FileNotFoundError as e:
        log_event("audio:convert_error", str(e))
        print(f"❌ Could not launch ffmpeg at '{ffmpeg_path}': {e}")
        return None
    except subprocess.TimeoutExpired:
        log_event("audio:convert_timeout", {"src": src_path})
        print(f"❌ ffmpeg timed out converting {src_path}")
        return None
    except Exception as e:
        log_event("audio:convert_error", str(e))
        print(f"❌ ffmpeg conversion raised an exception: {e}")
        return None


def convert_wav_to_ogg_opus(src_wav: str):
    """Convert a WAV file to OGG/Opus so it can be sent as a native Telegram
    voice-note bubble via sendVoice. Needs ffmpeg on PATH."""
    if not FFMPEG_AVAILABLE:
        return None
    dst_path = os.path.splitext(src_wav)[0] + ".ogg"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src_wav, "-c:a", "libopus", "-b:a", "32k", dst_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 or not os.path.exists(dst_path):
            log_event("audio:opus_convert_failed", {"src": src_wav, "stderr": result.stderr[-500:]})
            return None
        return dst_path
    except Exception as e:
        log_event("audio:opus_convert_error", str(e))
        return None


def transcribe_wav_file(wav_path: str, vosk_model) -> str:
    """Run a full (non-streaming) 16kHz mono PCM16 WAV file through Vosk and
    return the transcribed text."""
    try:
        wf = wave.open(wav_path, "rb")
    except Exception as e:
        return f"[transcription error: could not open wav: {e}]"

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        wf.close()
        return "[transcription error: audio is not 16-bit mono PCM after conversion]"

    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(False)
    text_parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result.get("text"):
                text_parts.append(result["text"])
    final = json.loads(rec.FinalResult())
    if final.get("text"):
        text_parts.append(final["text"])
    wf.close()
    return " ".join(text_parts).strip()


def perform_transcribe_voice(path: str, vosk_model) -> str:
    """Tool: transcribe a downloaded Telegram voice/audio file to text using Vosk."""
    print(f"🎙️ Transcribing voice file: {path}")
    if not path:
        return "Error: no voice file path provided."
    if not _is_path_allowed(path):
        log_event("transcribe_voice:blocked", {"path": path})
        return f"Error: access to '{path}' is not permitted (outside the assistant's allowed file locations)."
    if not os.path.exists(path):
        return f"Error: voice file not found: {path}"

    wav_path = convert_to_wav_16k_mono(path)
    if not wav_path:
        return "Error: could not convert voice file to WAV. Is ffmpeg installed and on PATH?"

    transcript = transcribe_wav_file(wav_path, vosk_model)
    try:
        os.remove(wav_path)
    except Exception:
        pass

    if not transcript:
        return "(No speech was detected in the voice message.)"
    log_event("transcribe_voice:result", {"path": path, "transcript": transcript})
    return transcript


def perform_ingest_image(path: str) -> dict:
    """Tool: load an image from disk and base64-encode it so it can be attached
    to the next Ollama request via the message 'images' field (this is how a
    vision-capable Ollama model actually receives pixel data — a bare path
    string means nothing to the model itself)."""
    print(f"🖼️ Ingesting image: {path}")
    if not path:
        return {"ok": False, "error": "No image file path provided."}
    if not _is_path_allowed(path):
        log_event("ingest_image:blocked", {"path": path})
        return {"ok": False, "error": f"Access to '{path}' is not permitted (outside the assistant's allowed file locations)."}
    if not os.path.exists(path):
        return {"ok": False, "error": f"Image file not found: {path}"}
    try:
        with open(path, "rb") as f:
            img_bytes = f.read()
        if len(img_bytes) > 20 * 1024 * 1024:
            return {"ok": False, "error": "Image exceeds 20MB and was not loaded."}
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        log_event("ingest_image:success", {"path": path, "bytes": len(img_bytes)})
        return {"ok": True, "base64": b64}
    except Exception as e:
        log_event("ingest_image:error", str(e))
        return {"ok": False, "error": str(e)}


def perform_read_file(path: str, max_chars: int = 6000) -> str:
    """Tool: read a text-like file from disk and return its (possibly truncated)
    contents. Non-text files are reported but not dumped, since they'd just be
    binary noise to the model."""
    print(f"📄 Reading file: {path}")
    if not path:
        return "Error: no file path provided."
    if not _is_path_allowed(path):
        log_event("read_file:blocked", {"path": path})
        return f"Error: access to '{path}' is not permitted (outside the assistant's allowed file locations)."
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    ext = os.path.splitext(path)[1].lower()
    if ext not in TEXT_FILE_EXTENSIONS:
        return (f"File saved at {path} ({ext or 'unknown type'}). "
                "This isn't a plain-text type, so it can't be read directly — "
                "use shell_exec or code_exec if you need to process it.")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        truncated = content[:max_chars]
        suffix = "\n\n[...truncated...]" if len(content) > max_chars else ""
        return truncated + suffix
    except Exception as e:
        return f"Error reading file: {e}"
