"""Telegrambot.py bridge: outbound send helpers, the long-polling inbound listener,
and turning a queued Telegram message into what the model sees this turn."""
import os
import queue
import threading
import time

import requests

from jarvis.config import CONFIG
from jarvis.logging_utils import log_event
from jarvis.tools.media import perform_transcribe_voice, perform_ingest_image

TELEGRAM_BOT_TOKEN = CONFIG.telegram_bot_token
TELEGRAM_CHAT_ID = CONFIG.telegram_chat_id
TELEGRAM_POLL_TIMEOUT = CONFIG.get("telegram", "poll_timeout", default=30)
STREAM_TOOL_CALLS = CONFIG.get("telegram", "stream_tool_calls", default=True)

TELEGRAM_IMAGES_DIR = CONFIG.get("telegram", "images_dir", default="telegram_images")
TELEGRAM_AUDIO_DIR = CONFIG.get("telegram", "audio_dir", default="telegram_audio")
TELEGRAM_FILES_DIR = CONFIG.get("telegram", "files_dir", default="telegram_files")
TELEGRAM_VOICE_OUT_DIR = CONFIG.get("telegram", "voice_out_dir", default="telegram_voice_replies")

for _dir in (TELEGRAM_IMAGES_DIR, TELEGRAM_AUDIO_DIR, TELEGRAM_FILES_DIR, TELEGRAM_VOICE_OUT_DIR):
    os.makedirs(_dir, exist_ok=True)

telegram_queue = queue.Queue()  # for receiving Telegram messages in main thread
telegram_thread = None
telegram_thread_running = False


def telegram_get_file_path(file_id: str):
    """Ask Telegram for the server-side path of a file_id (needed before downloading it)."""
    if not TELEGRAM_BOT_TOKEN:
        return None
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"
        resp = requests.get(url, params={"file_id": file_id}, timeout=10)
        if resp.ok:
            data = resp.json()
            if data.get("ok"):
                return data["result"]["file_path"]
        log_event("telegram:getFile_failed", {"status_code": resp.status_code, "text": resp.text})
    except Exception as e:
        log_event("telegram:getFile_error", str(e))
    return None


def telegram_download_file(file_id: str, dest_dir: str, suffix_hint: str = "") -> str:
    """Download a Telegram file by file_id and save it timestamped into dest_dir.
    Returns the local path, or None on failure. Note: Telegram's Bot API caps
    downloadable file size at 20MB unless you're running a local Bot API server."""
    remote_path = telegram_get_file_path(file_id)
    if not remote_path:
        return None
    try:
        url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{remote_path}"
        resp = requests.get(url, timeout=30)
        if not resp.ok:
            log_event("telegram:download_failed", {"status_code": resp.status_code})
            return None
        ext = os.path.splitext(remote_path)[1] or suffix_hint or ""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{file_id[:8]}{ext}"
        local_path = os.path.join(dest_dir, filename)
        with open(local_path, "wb") as f:
            f.write(resp.content)
        log_event("telegram:file_downloaded", {"file_id": file_id, "local_path": local_path, "bytes": len(resp.content)})
        return local_path
    except Exception as e:
        log_event("telegram:download_error", str(e))
        return None


def telegram_send_voice(chat_id: str, file_path: str) -> bool:
    """Send an OGG/Opus file as a native Telegram voice-note bubble."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVoice"
        with open(file_path, "rb") as f:
            resp = requests.post(url, data={"chat_id": chat_id}, files={"voice": f}, timeout=30)
        if resp.ok:
            return True
        log_event("telegram:send_voice_failed", {"status_code": resp.status_code, "text": resp.text})
    except Exception as e:
        log_event("telegram:send_voice_error", str(e))
    return False


def telegram_send_audio(chat_id: str, file_path: str) -> bool:
    """Fallback for when ffmpeg isn't available to produce OGG/Opus: sends the
    reply as a regular audio file attachment instead of a voice-note bubble."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendAudio"
        with open(file_path, "rb") as f:
            resp = requests.post(url, data={"chat_id": chat_id}, files={"audio": f}, timeout=30)
        if resp.ok:
            return True
        log_event("telegram:send_audio_failed", {"status_code": resp.status_code, "text": resp.text})
    except Exception as e:
        log_event("telegram:send_audio_error", str(e))
    return False


def telegram_send_photo(chat_id: str, file_path: str, caption: str = "") -> bool:
    """Send an image file as a native Telegram photo bubble. Used by cam_peek
    to push the captured webcam frame to the user when they're remote."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(file_path, "rb") as f:
            data = {"chat_id": chat_id}
            if caption:
                data["caption"] = caption
            resp = requests.post(url, data=data, files={"photo": f}, timeout=30)
        if resp.ok:
            return True
        log_event("telegram:send_photo_failed", {"status_code": resp.status_code, "text": resp.text})
    except Exception as e:
        log_event("telegram:send_photo_error", str(e))
    return False


def telegram_stream_status(chat_id: str, text: str) -> None:
    """Send a short, best-effort status update to Telegram (e.g. 'searching the
    web...'). Silently does nothing if streaming is disabled, there's no
    chat_id (non-Telegram turn), or the send fails — this is a nice-to-have,
    never something a turn should fail over."""
    if not STREAM_TOOL_CALLS or not chat_id:
        return
    try:
        telegram_send_message(chat_id, text)
    except Exception as e:
        log_event("telegram:tool_stream_error", str(e))


def telegram_try_dequeue_authorized(block: bool = False, timeout: float = 1.0):
    """Pop the next message from telegram_queue, if any, and drop it unless it
    comes from the configured TELEGRAM_CHAT_ID. This is the single gate every
    call site should go through — never read telegram_queue directly, or an
    unauthorized chat can slip through with no filtering at all.

    block=False: non-blocking check (used when polling alongside something
    else, e.g. mic input in the speech-mode listen loop).
    block=True: waits up to `timeout` seconds for a message (used by the
    dedicated Telegram-mode loop in main.py).

    Returns the message dict, or None if there was nothing to process (queue
    empty, or the message was from an unauthorized chat and got dropped).
    """
    try:
        tg_msg = telegram_queue.get(timeout=timeout) if block else telegram_queue.get_nowait()
    except queue.Empty:
        return None

    chat_id = tg_msg.get("chat_id")
    if TELEGRAM_CHAT_ID and str(TELEGRAM_CHAT_ID) != str(chat_id):
        log_event("telegram:ignored", {"chat_id": chat_id, "reason": "unauthorized_chat_id"})
        return None
    return tg_msg


def telegram_send_message(chat_id: str, text: str) -> bool:
    """Send a text message via Telegram Bot API. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        resp = requests.post(url, json=payload, timeout=8)
        if resp.ok:
            return True
        log_event("telegram:send_failed", {"status_code": resp.status_code, "text": resp.text})
    except Exception as e:
        log_event("telegram:send_error", str(e))
    return False


def _notify_download_failure(chat_id: str, kind: str) -> None:
    """Tell the user their file didn't come through, instead of the update
    just vanishing. Without this, a failed getFile/download for a voice
    note, photo, or document silently drops the update (the offset has
    already advanced past it) and the user is left waiting on a message
    that will never get a reply."""
    log_event("telegram:download_failed_notify", {"chat_id": chat_id, "kind": kind})
    try:
        telegram_send_message(chat_id, f"⚠️ I couldn't download that {kind} — please try sending it again.")
    except Exception as e:
        log_event("telegram:download_failed_notify_error", str(e))


def _telegram_polling_loop():
    """Background thread target: long-poll Telegram getUpdates and enqueue new text messages."""
    global telegram_thread_running
    if not TELEGRAM_BOT_TOKEN:
        log_event("telegram:disabled", "No TELEGRAM_BOT_TOKEN provided")
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    get_updates_url = base_url + "/getUpdates"
    offset = None
    backoff = 1.0
    while telegram_thread_running:
        try:
            params = {"timeout": TELEGRAM_POLL_TIMEOUT}
            if offset:
                params["offset"] = offset
            resp = requests.get(get_updates_url, params=params, timeout=TELEGRAM_POLL_TIMEOUT + 10)
            if not resp.ok:
                log_event("telegram:getUpdates_failed", {"status_code": resp.status_code, "text": resp.text})
                time.sleep(min(10, backoff))
                backoff = min(10, backoff * 1.5)
                continue

            data = resp.json()
            backoff = 1.0
            if not data.get("ok"):
                time.sleep(1.0)
                continue

            updates = data.get("result", [])
            for upd in updates:
                update_id = upd.get("update_id")
                if update_id is None:
                    continue
                offset = update_id + 1

                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat = msg.get("chat", {})
                chat_id = str(chat.get("id"))
                from_id = str(msg.get("from", {}).get("id", ""))
                caption = msg.get("caption")

                queued_item = None

                if msg.get("voice"):
                    file_id = msg["voice"]["file_id"]
                    local_path = telegram_download_file(file_id, TELEGRAM_AUDIO_DIR, suffix_hint=".oga")
                    if local_path:
                        queued_item = {"type": "voice", "path": local_path, "caption": caption}
                    else:
                        _notify_download_failure(chat_id, "voice message")
                elif msg.get("audio"):
                    file_id = msg["audio"]["file_id"]
                    local_path = telegram_download_file(file_id, TELEGRAM_AUDIO_DIR, suffix_hint=".mp3")
                    if local_path:
                        queued_item = {"type": "voice", "path": local_path, "caption": caption}
                    else:
                        _notify_download_failure(chat_id, "audio message")
                elif msg.get("photo"):
                    largest = msg["photo"][-1]
                    file_id = largest["file_id"]
                    local_path = telegram_download_file(file_id, TELEGRAM_IMAGES_DIR, suffix_hint=".jpg")
                    if local_path:
                        queued_item = {"type": "image", "path": local_path, "caption": caption}
                    else:
                        _notify_download_failure(chat_id, "photo")
                elif msg.get("document"):
                    doc = msg["document"]
                    file_id = doc["file_id"]
                    mime = doc.get("mime_type", "") or ""
                    orig_name = doc.get("file_name", "") or ""
                    ext = os.path.splitext(orig_name)[1]
                    if mime.startswith("image/"):
                        local_path = telegram_download_file(file_id, TELEGRAM_IMAGES_DIR, suffix_hint=ext or ".jpg")
                        if local_path:
                            queued_item = {"type": "image", "path": local_path, "caption": caption}
                        else:
                            _notify_download_failure(chat_id, "image")
                    else:
                        local_path = telegram_download_file(file_id, TELEGRAM_FILES_DIR, suffix_hint=ext)
                        if local_path:
                            queued_item = {"type": "file", "path": local_path, "caption": caption}
                        else:
                            _notify_download_failure(chat_id, "file")
                else:
                    text = msg.get("text") or msg.get("caption")
                    if text:
                        queued_item = {"type": "text", "text": text}

                if not queued_item:
                    continue

                queued_item.update({"chat_id": chat_id, "from_id": from_id, "update_id": update_id})
                telegram_queue.put(queued_item)
                log_event("telegram:enqueued", {
                    "chat_id": chat_id, "from_id": from_id, "update_id": update_id,
                    "type": queued_item["type"], "detail": queued_item.get("text", queued_item.get("path", ""))[:120]
                })

        except Exception as e:
            log_event("telegram:poll_error", str(e))
            time.sleep(min(10, backoff))
            backoff = min(10, backoff * 1.5)


def start_telegram_listener():
    """Start the Telegram polling thread if token is available and the feature is enabled."""
    global telegram_thread, telegram_thread_running
    if not CONFIG.feature_enabled("telegram"):
        print("Telegram feature disabled in config.json; skipping listener.")
        return
    if not TELEGRAM_BOT_TOKEN:
        print("Telegram bot token not configured; skipping Telegram listener.")
        return
    if telegram_thread and telegram_thread.is_alive():
        return
    telegram_thread_running = True
    telegram_thread = threading.Thread(target=_telegram_polling_loop, daemon=True)
    telegram_thread.start()
    log_event("telegram:listener_started")


def stop_telegram_listener():
    """Stop the Telegram polling thread gracefully."""
    global telegram_thread, telegram_thread_running
    telegram_thread_running = False
    if telegram_thread and telegram_thread.is_alive():
        telegram_thread.join(timeout=5)
    telegram_thread = None
    log_event("telegram:listener_stopped")


def prepare_telegram_turn(tg_msg: dict, vosk_model):
    """Convert a queued Telegram message into what the model actually sees this turn:
    (user_input_text, reply_with_voice, attached_images).

    Voice messages are transcribed immediately via Vosk — exactly like local mic input
    already works — so the model just receives plain text, no tool call needed.
    Images are loaded and base64-encoded immediately so they can be attached directly
    to the first Ollama call for this turn, no round trip needed either.

    Files are still handled via the read_file tool, since reading a document is a
    genuine choice the model should make rather than something forced on every file.
    """
    msg_type = tg_msg.get("type", "text")
    caption = (tg_msg.get("caption") or "").strip()

    if msg_type == "voice":
        path = tg_msg.get("path", "")
        if vosk_model is None:
            return "(A voice message arrived, but speech recognition isn't available right now — let the user know.)", True, None
        transcript = perform_transcribe_voice(path, vosk_model)
        if caption:
            transcript = f"{transcript}\n\n(Caption: {caption})"
        return transcript, True, None

    if msg_type == "image":
        path = tg_msg.get("path", "")
        image_result = perform_ingest_image(path)
        if not image_result.get("ok"):
            return f"(An image arrived but couldn't be loaded: {image_result.get('error')})", False, None
        text = caption if caption else "Here's an image I just sent — take a look and respond to it."
        return text, False, [image_result["base64"]]

    if msg_type == "file":
        path = tg_msg.get("path", "")
        note = f"[User sent a file. File path: {path}."
        if caption:
            note += f" Caption: {caption}."
        note += " Use the read_file tool on this file path if you need to see its contents.]"
        return note, False, None

    return (tg_msg.get("text") or "").strip(), False, None
