"""Entry point. Run with: python main.py

Loads config.json + .env, initializes speech/TTS/Ollama, starts the Telegram
listener and DJ auto-advance monitor (if enabled), then runs the main
listen -> think -> speak loop until 'q' is pressed or Ctrl+C.
"""
import time

import keyboard
import pyttsx3

from jarvis.audio_io import (
    configure_tts_engine,
    handle_controls,
    listen_and_transcribe,
    load_tts_settings,
    load_vosk_model,
    speak_text,
    synthesize_voice_reply,
)
from jarvis.capabilities import CV2_AVAILABLE, YOLO_AVAILABLE, YTDLP_AVAILABLE, YTDLP_PATH, diagnose_ffmpeg
from jarvis.config import CONFIG
from jarvis.logging_utils import log_event, log_interaction
from jarvis.memory import load_conversation_history, maybe_run_memory_maintenance, save_conversation_history
from jarvis.ollama_client import check_ollama_running, query_ollama_with_web
from jarvis.session import (
    handle_autopilot_command,
    handle_reset_command,
    run_autopilot_session,
    toggle_autopilot_mode,
    toggle_input_mode,
    toggle_telegram_mode,
)
from jarvis.soul import inject_soul_context
from jarvis.system_prompt import build_system_prompt
from jarvis.telegram_bot import (
    telegram_send_audio,
    telegram_send_message,
    telegram_send_voice,
    prepare_telegram_turn,
    start_telegram_listener,
    stop_telegram_listener,
    telegram_try_dequeue_authorized,
)
from jarvis.tools.dj_mode import dj_shutdown, start_dj_monitor
from jarvis.tools.web_search import get_last_search

DEBUG_PATH = CONFIG.get("debug", "path", default=False)
DEBUG_DJ = CONFIG.get("debug", "dj", default=False)
DEBUG_CAM = CONFIG.get("debug", "cam", default=False)
DEBUG_TTS = CONFIG.get("debug", "tts", default=False)


def main():
    try:
        log_event("session:start")
        model = load_vosk_model()
        if DEBUG_TTS:
            print("Initializing TTS engine...")

        tts_engine = pyttsx3.init()
        tts_settings = load_tts_settings()
        configure_tts_engine(tts_engine, tts_settings)
        if DEBUG_TTS:
            print(f"TTS engine initialized with settings: {tts_settings}")

        log_event("tts:initialized", tts_settings)

        check_ollama_running()

        if DEBUG_PATH:
            diagnose_ffmpeg()

        if DEBUG_DJ:
            if YTDLP_AVAILABLE:
                print(f"yt-dlp found: {YTDLP_PATH} (DJ mode enabled)")
            else:
                print("⚠️  yt-dlp NOT found on PATH — DJ mode (dj_play/dj_skip/dj_stop) will not work until you `pip install yt-dlp`.")

        if DEBUG_CAM:
            if CV2_AVAILABLE and YOLO_AVAILABLE:
                print("opencv-python + ultralytics found (cam_peek enabled)")
            else:
                missing = [pkg for pkg, ok in (("opencv-python", CV2_AVAILABLE), ("ultralytics", YOLO_AVAILABLE)) if not ok]
                print(f"⚠️  Missing {', '.join(missing)} — cam_peek will not work until you `pip install {' '.join(missing)}`.")

        # ==================== PERSISTENT MEMORY + SYSTEM PROMPT ====================
        history_file = CONFIG.get("files", "conversation_history", default="conversation_history.json")
        messages = load_conversation_history(history_file)
        messages = inject_soul_context(messages, inject_mode="optional")

        system_prompt = build_system_prompt()
        version_tag = f"[PROMPT_VERSION:{CONFIG.get('assistant', 'system_prompt_version', default='jarvis_v1.8')}]"

        system_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "system"]
        if not system_indices:
            messages.insert(0, {"role": "system", "content": system_prompt})
            save_conversation_history(messages, history_file)
        else:
            primary_idx = system_indices[0]
            current_system = messages[primary_idx].get("content", "")
            if version_tag not in current_system:
                messages[primary_idx] = {"role": "system", "content": system_prompt}
                for idx in reversed(system_indices[1:]):
                    messages.pop(idx)
                log_event("history:system_prompt_refreshed", {"version": version_tag})
                save_conversation_history(messages, history_file)

        state = {
            "bot_muted": False,
            "mic_muted": False,
            "stop": False,
            "input_mode": "speech",
            "assistant_mode": "chat",
            "autopilot": False,
            "autopilot_turns": 0,
            "autopilot_goal": "",
            "memory_prompt_count": 0,
            "memory_warning_announced": False,
            "pending_reset_confirm": None,
            "vosk_model": model,
        }
        if keyboard:
            keyboard.add_hotkey("ctrl+shift+k", lambda: toggle_input_mode(state))
            keyboard.add_hotkey("ctrl+shift+a", lambda: toggle_autopilot_mode(state))
            keyboard.add_hotkey("ctrl+shift+t", lambda: toggle_telegram_mode(state))
        else:
            print("Install the 'keyboard' package to enable input mode hotkeys.")
        print("Controls: 'm' = bot mute, 'i' = mic mute, 'q' = quit, 'Ctrl+Shift+K' = toggle input device, 'Ctrl+Shift+A' = toggle autopilot, 'Ctrl+Shift+T' = Telegram mode\n")

        try:
            start_telegram_listener()
        except Exception as e:
            log_event("telegram:start_error", str(e))
        try:
            start_dj_monitor()
        except Exception as e:
            log_event("dj:monitor_start_error", str(e))

        while True:
            if state["stop"]:
                break

            if state.get("autopilot"):
                autopilot_step = run_autopilot_session(messages, state, tts_engine, history_file)
                if state["stop"]:
                    break
                time.sleep(0.5)
                continue

            telegram_origin = False
            telegram_reply_voice = False
            attached_images = None
            tg_msg = None
            chat_id = None

            if state.get("input_mode") == "telegram":
                print("Input mode: telegram — waiting for messages. Press Ctrl+Shift+T to toggle.")
                while state.get("input_mode") == "telegram" and not state.get("stop"):
                    tg_msg = telegram_try_dequeue_authorized(block=True, timeout=1)
                    if tg_msg:
                        break
                    handle_controls(state)
                if tg_msg:
                    chat_id = tg_msg.get("chat_id")
                    user_input, telegram_reply_voice, attached_images = prepare_telegram_turn(tg_msg, state.get("vosk_model"))
                    telegram_origin = True
                    print(f"[telegram:{tg_msg.get('type', 'text')}] {user_input}\n")
                    log_interaction(user_input, "(from telegram)", assistant_mode=state.get("assistant_mode"))
                    try:
                        telegram_send_message(chat_id, "⏳")
                    except Exception:
                        pass

            if not tg_msg:
                tg_msg = telegram_try_dequeue_authorized(block=False)

            if tg_msg and not telegram_origin:
                chat_id = tg_msg.get("chat_id")
                user_input, telegram_reply_voice, attached_images = prepare_telegram_turn(tg_msg, state.get("vosk_model"))
                telegram_origin = True
                print(f"[telegram:{tg_msg.get('type', 'text')}] {user_input}\n")
                log_interaction(user_input, "(from telegram)", assistant_mode=state.get("assistant_mode"))
                try:
                    telegram_send_message(chat_id, "⏳")
                except Exception:
                    pass

            if not telegram_origin:
                if state["input_mode"] == "speech":
                    result = listen_and_transcribe(model, state)
                    if state["stop"]:
                        break
                    if not result:
                        continue
                    if isinstance(result, dict):
                        # An authorized Telegram message arrived while listening on the mic.
                        chat_id = result["chat_id"]
                        user_input = result["text"]
                        telegram_reply_voice = result["reply_voice"]
                        attached_images = result["attached_images"]
                        telegram_origin = True
                        print(f"[telegram:mic-interrupt] {user_input}\n")
                        log_interaction(user_input, "(from telegram)", assistant_mode=state.get("assistant_mode"))
                    else:
                        user_input = result
                else:
                    handle_controls(state)
                    if state["stop"]:
                        break
                    user_input = input("Jarvis keyboard mode > ").strip()
                    if not user_input:
                        continue

            print(f"You said: {user_input}\n")

            autopilot_response = handle_autopilot_command(state, user_input)
            reset_response = None
            if autopilot_response is None:
                reset_response = handle_reset_command(state, user_input, messages, history_file, system_prompt)
            local_command_response = autopilot_response if autopilot_response is not None else reset_response

            if local_command_response is not None:
                print(f"Jarvis: {local_command_response}\n")
                log_interaction(
                    user_input,
                    local_command_response,
                    autopilot_turns=state.get("autopilot_turns"),
                    assistant_mode=state.get("assistant_mode"),
                )
                log_event("interaction:logged")
                save_conversation_history(messages, history_file)
                try:
                    if telegram_origin:
                        telegram_send_message(chat_id, local_command_response)
                except Exception as e:
                    log_event("telegram:reply_send_error", str(e))
                if not state["bot_muted"]:
                    speak_text(tts_engine, local_command_response, state)
                continue

            messages, warning_text, memory_note, _ = maybe_run_memory_maintenance(messages, state, history_file, incoming_text=user_input)
            if warning_text:
                print(warning_text)
                log_event("memory:user_warning", {"source": "main", "warning": warning_text})

            runtime_state = dict(state)
            runtime_state["memory_note"] = memory_note
            runtime_state["vosk_model"] = model
            runtime_state["telegram_chat_id"] = chat_id if telegram_origin else None

            response = query_ollama_with_web(messages, user_input, runtime_state, attached_images=attached_images)
            if warning_text:
                response = f"{warning_text} {response}".strip()
            print(f"Jarvis: {response}\n")
            last_search = get_last_search()
            log_interaction(
                user_input,
                response,
                search_query=last_search.get("query"),
                search_results=last_search.get("results"),
                assistant_mode=state.get("assistant_mode"),
            )
            log_event("interaction:logged")
            save_conversation_history(messages, history_file)

            # If the prompt originated from Telegram, send the assistant response back there.
            # Voice-origin turns get a synthesized voice-note reply; everything else gets text.
            if telegram_origin:
                try:
                    if telegram_reply_voice:
                        voice_reply_path = synthesize_voice_reply(response, tts_settings)
                        sent = False
                        if voice_reply_path and voice_reply_path.endswith(".ogg"):
                            sent = telegram_send_voice(chat_id, voice_reply_path)
                        elif voice_reply_path:
                            sent = telegram_send_audio(chat_id, voice_reply_path)
                        if not sent:
                            telegram_send_message(chat_id, response)
                    else:
                        telegram_send_message(chat_id, response)
                except Exception as e:
                    log_event("telegram:reply_send_error", str(e))

            if state["stop"]:
                break

            time.sleep(0.5)

            if state["stop"]:
                break

            if not state["bot_muted"]:
                speak_text(tts_engine, response, state)

            if state["stop"]:
                break

        try:
            stop_telegram_listener()
        except Exception:
            pass
        try:
            dj_shutdown()
        except Exception:
            pass
    except KeyboardInterrupt:
        log_event("session:keyboard_interrupt")
        print("\nExiting. Goodbye!")
        try:
            dj_shutdown()
        except Exception:
            pass
    except Exception as e:
        try:
            stop_telegram_listener()
        except Exception:
            pass
        try:
            dj_shutdown()
        except Exception:
            pass
        log_event("session:error", str(e))
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
