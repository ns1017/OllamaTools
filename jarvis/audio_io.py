"""Microphone input (Vosk), spoken output (pyttsx3), the outgoing-voice-reply
synthesizer used for Telegram voice notes, and the keyboard mute/quit
controls shared by both the listen and speak loops."""
import json
import os
import time

import msvcrt
import pyttsx3
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from jarvis.config import CONFIG
from jarvis.logging_utils import log_event
from jarvis.telegram_bot import prepare_telegram_turn, telegram_send_message, telegram_try_dequeue_authorized
from jarvis.tools.media import convert_wav_to_ogg_opus

TTS_DEFAULTS = {"rate": CONFIG.get("tts", "rate", default=155), "volume": CONFIG.get("tts", "volume", default=0.7)}
TTS_DEBUG = CONFIG.get("debug", "tts", default=False)


def load_vosk_model():
    model_path = CONFIG.get("vosk", "model_path", default="models/vosk-model-small-en-us-0.15")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Vosk model not found!")
    return Model(model_path)


def handle_controls(state):
    """Keyboard-driven mute/quit controls, polled during listening and speaking."""
    if not msvcrt.kbhit():
        return
    key = msvcrt.getwch().lower()
    if key == "m":
        state["bot_muted"] = not state["bot_muted"]
        print("Bot muted" if state["bot_muted"] else "Bot unmuted")
    elif key == "i":
        state["mic_muted"] = not state["mic_muted"]
        print("Mic muted" if state["mic_muted"] else "Mic unmuted")
    elif key == "q":
        state["stop"] = True


def listen_and_transcribe(model, state):
    """Blocks on the mic, but also polls for an incoming (authorized) Telegram
    message on every cycle so the assistant can respond to Telegram promptly
    even while sitting in speech-input mode.

    Returns:
        - None if state["stop"] was set (caller should exit).
        - "" if there's nothing to act on yet but the caller should re-poll
          (e.g. input_mode changed away from speech).
        - a plain str with the transcribed text, for ordinary mic input.
        - a dict — {"text", "telegram_origin": True, "chat_id", "reply_voice",
          "attached_images"} — when an authorized Telegram message arrived
          while listening. Callers must check `isinstance(result, dict)` to
          tell this case apart from plain mic text.
    """
    rec = KaldiRecognizer(model, 16000)
    with sd.RawInputStream(samplerate=16000, blocksize=4096, dtype="int16", channels=1) as stream:
        print("Listening... Speak now.\n")
        while True:
            tg_msg = telegram_try_dequeue_authorized(block=False)
            if tg_msg:
                chat_id = tg_msg.get("chat_id")
                text, reply_voice, attached_images = prepare_telegram_turn(tg_msg, model)
                log_event("telegram:dequeued_in_listen", {"chat_id": chat_id, "type": tg_msg.get("type", "text"), "text": text[:120]})
                try:
                    telegram_send_message(chat_id, "⏳")
                except Exception:
                    pass
                return {
                    "text": text,
                    "telegram_origin": True,
                    "chat_id": chat_id,
                    "reply_voice": reply_voice,
                    "attached_images": attached_images,
                }
            handle_controls(state)
            if state["stop"]:
                return None
            if state.get("input_mode") != "speech":
                return ""
            data, overflowed = stream.read(4096)
            if overflowed:
                continue
            if state["mic_muted"]:
                continue
            chunk = bytes(data)
            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())
                text = result.get('text', '')
                if text:
                    return text


def speak_text(engine, text, state):
    try:
        if state.get("bot_muted") or state.get("stop"):
            return

        engine.stop()
        engine.say(text)

        loop_started = False
        try:
            engine.startLoop(False)
            loop_started = True
            while engine.isBusy():
                handle_controls(state)
                if state.get("stop") or state.get("bot_muted"):
                    engine.stop()
                    break
                engine.iterate()
        finally:
            if loop_started:
                engine.endLoop()

    except Exception as e:
        print(f"Error speaking text: {e}")


def load_tts_settings(path: str = None):
    path = path or CONFIG.get("files", "tts_settings", default="tts_settings.json")
    if not os.path.exists(path):
        return dict(TTS_DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as handler:
            loaded = json.load(handler)
            if not isinstance(loaded, dict):
                raise ValueError("TTS settings file must contain a JSON object.")
            return {**TTS_DEFAULTS, **loaded}
    except Exception as err:
        print(f"Warning: Could not load TTS settings: {err}")
        return dict(TTS_DEFAULTS)


def configure_tts_engine(engine, settings):
    target_voice = settings.get("voice")
    if target_voice:
        chosen = None
        for voice in engine.getProperty("voices"):
            if target_voice.lower() in voice.name.lower():
                chosen = voice
                break
        if chosen:
            engine.setProperty("voice", chosen.id)
            print(f"TTS voice set to {chosen.name}")
        else:
            print(f"Warning: Voice '{target_voice}' not found. Using default voice.")

    rate = settings.get("rate")
    if isinstance(rate, (int, float)):
        engine.setProperty("rate", int(rate))

    volume = settings.get("volume")
    if isinstance(volume, (int, float)):
        engine.setProperty("volume", max(0.0, min(1.0, float(volume))))

    pitch = settings.get("pitch")
    if isinstance(pitch, (int, float)):
        try:
            engine.setProperty("pitch", int(pitch))
        except Exception:
            print("Info: Pitch control not supported by current driver.")


def synthesize_voice_reply(text: str, tts_settings, out_dir: str = None):
    """Render text to speech with pyttsx3 and return a path to an OGG/Opus file
    ready for sendVoice. Falls back to a plain WAV path (for sendAudio) if
    ffmpeg isn't available to do the OGG/Opus conversion."""
    out_dir = out_dir or CONFIG.get("telegram", "voice_out_dir", default="telegram_voice_replies")
    if not text or not text.strip():
        return None
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    wav_path = os.path.join(out_dir, f"reply_{timestamp}.wav")
    try:
        # Dedicated engine instance so this doesn't collide with the main
        # loop's tts_engine, which may be mid-speech on the mic side.
        engine = pyttsx3.init()
        configure_tts_engine(engine, tts_settings)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
    except Exception as e:
        log_event("tts:save_to_file_error", str(e))
        return None

    if not os.path.exists(wav_path):
        return None

    ogg_path = convert_wav_to_ogg_opus(wav_path)
    if ogg_path:
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return ogg_path

    log_event("tts:ogg_fallback", "ffmpeg unavailable; sending WAV via sendAudio instead of sendVoice")
    return wav_path
