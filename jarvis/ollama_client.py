"""ollama_client.py Talks to the local Ollama server and runs the tool-calling loop: detect a
tool tag in the model's response, execute it, feed the result back, repeat
until the model produces a final answer (or a safety limit is hit)."""
import json
import re
import subprocess
import time

import requests

from jarvis.config import CONFIG
from jarvis.logging_utils import log_event
from jarvis.memory import trim_history
from jarvis.soul import append_soul_entry
from jarvis.telegram_bot import telegram_stream_status
from jarvis.text_utils import (
    build_runtime_context,
    detect_peek_typo_hint,
    extract_search_mode,
    extract_tool_request,
    guard_against_unverified_visual_claims,
    sanitize_response,
)
from jarvis.tools.dj_mode import (
    perform_dj_play,
    perform_dj_queue_list,
    perform_dj_queue_remove,
    perform_dj_skip,
    perform_dj_stop,
)
from jarvis.tools.gmail import perform_gmail_check
from jarvis.tools.media import perform_ingest_image, perform_read_file, perform_transcribe_voice
from jarvis.tools.school_calendar import perform_school_calendar
from jarvis.tools.shell import perform_code_dev, perform_code_exec, perform_get_environment, perform_shell_exec
from jarvis.tools.vision import perform_cam_peek, perform_peek
from jarvis.tools.web_search import get_last_search, search_with_cache

OLLAMA_DEBUG = CONFIG.get("debug", "ollama", default=False)


def check_ollama_running():
    """Check if the Ollama server is running by making a simple request; try
    to start it if it's not."""
    base_url = CONFIG.get("ollama", "base_url", default="http://127.0.0.1:11434")
    try:
        response = requests.get(f"{base_url}/api/chat", timeout=2)
        if OLLAMA_DEBUG:
            print(f"Ollama server check: {response.status_code}")
        if response.status_code == 405:
            print('Started Successfully!')  # Method Not Allowed is expected here, means server is up
            return True
    except requests.exceptions.ConnectionError:
        print("Ollama server not responding, attempting to start it...")
        try:
            subprocess.Popen("ollama serve", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Ollama server starting in background...")
            time.sleep(5)
            print("Checking if server is now available...")
        except Exception as e:
            print(f"Failed to start Ollama server: {e}")


def generate_ollama_response(messages, runtime_context=None):
    """Helper: get a response from Ollama (streaming)."""
    send_messages = trim_history(messages)
    if runtime_context:
        runtime_message = {"role": "system", "content": runtime_context}
        if send_messages and send_messages[0].get("role") == "system":
            send_messages.insert(1, runtime_message)
        else:
            send_messages.insert(0, runtime_message)

    base_url = CONFIG.get("ollama", "base_url", default="http://127.0.0.1:11434")
    url = f"{base_url}/api/chat"
    payload = {
        "model": CONFIG.get("ollama", "model"),
        "messages": send_messages,
        "stream": True,
        "options": {"temperature": CONFIG.get("ollama", "temperature", default=0.7)},
    }

    log_event("ollama:request", {"model": payload["model"], "messages": len(send_messages)})

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(CONFIG.get("ollama", "connect_timeout", default=10), CONFIG.get("ollama", "read_timeout", default=45)),
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "message" in data and "content" in data["message"]:
                    full_response += data["message"]["content"]
                if data.get("done", False):
                    break
        log_event("ollama:response", full_response)
        return full_response.strip()
    except requests.exceptions.Timeout:
        timeout_msg = "Error communicating with Ollama: response stream timed out."
        log_event("ollama:error", timeout_msg)
        return timeout_msg
    except Exception as e:
        log_event("ollama:error", str(e))
        return f"Error communicating with Ollama: {e}"


def _enabled_tags() -> tuple:
    """Which tool tags the loop should actually watch for this run, based on
    config.json -> features. Disabled tools are simply never dispatched even
    if a stray tag shows up in the text (it gets stripped by sanitize_response
    at the end either way)."""
    tags = ["continue"]
    if CONFIG.feature_enabled("web_search"):
        tags.append("web_search")
    if CONFIG.feature_enabled("shell_exec"):
        tags.append("shell_exec")
    if CONFIG.feature_enabled("code_tools"):
        tags += ["code_exec", "code_dev"]
    if CONFIG.feature_enabled("environment_tool"):
        tags.append("get_environment")
    if CONFIG.feature_enabled("peek"):
        tags.append("peek")
    if CONFIG.feature_enabled("cam_peek"):
        tags.append("cam_peek")
    if CONFIG.feature_enabled("school_calendar"):
        tags.append("school_calendar")
    if CONFIG.feature_enabled("gmail"):
        tags.append("gmail")
    if CONFIG.feature_enabled("soul_memory"):
        tags.append("soul_write")
    if CONFIG.feature_enabled("dj_mode"):
        tags += ["dj_play", "dj_stop", "dj_skip", "dj_queue_remove", "dj_queue_list"]
    # Generic media/file tools stay available regardless of the telegram feature toggle.
    tags += ["transcribe_voice", "ingest_image", "read_file"]
    return tuple(tags)


def query_ollama_with_web(messages, user_input, runtime_state=None, attached_images=None):
    runtime_state = runtime_state or {}
    telegram_chat_id = runtime_state.get("telegram_chat_id")
    turn_source = "telegram" if telegram_chat_id else "host"

    max_tool_iterations = CONFIG.get("turns", "max_tool_iterations", default=12)
    max_autonomous_turns = CONFIG.get("turns", "max_autonomous_turns", default=12)

    user_content = f"[source: {turn_source}] {user_input}"
    peek_hint = detect_peek_typo_hint(user_input)
    if peek_hint:
        user_content += f"\n\n[{peek_hint}]"
        log_event("peek:typo_hint_injected", {"user_input": user_input, "hint": peek_hint})

    user_message = {"role": "user", "content": user_content}
    if attached_images:
        user_message["images"] = attached_images
    messages.append(user_message)

    log_event("turn:start", {"user_input": user_input, "source": turn_source, "has_images": bool(attached_images)})

    assistant_mode = runtime_state.get("assistant_mode", "chat")
    autopilot_enabled = bool(runtime_state.get("autopilot", False))
    autopilot_goal = runtime_state.get("autopilot_goal", "")
    # This is the actual session-level counter session.py's run_autopilot_session
    # enforces the autopilot_max_turns cap against — distinct from the
    # tool-loop's local `autonomous_turns` below, which only counts <continue>
    # calls within this single turn and resets to 0 every time. Using the
    # latter for the budget line told the model it had a fresh budget every
    # turn right up until it got cut off.
    autopilot_turns_used = int(runtime_state.get("autopilot_turns", 0) or 0)
    memory_note = runtime_state.get("memory_note", "")

    def _stream(text: str) -> None:
        telegram_stream_status(telegram_chat_id, text)

    response_text = generate_ollama_response(
        messages,
        runtime_context=build_runtime_context(
            autonomous_turns=0,
            tool_iterations=0,
            autopilot_enabled=autopilot_enabled,
            autopilot_goal=autopilot_goal,
            autopilot_turns_used=autopilot_turns_used,
            memory_note=memory_note,
            mode=assistant_mode,
        ),
    )
    log_event("llm:raw_response", response_text)
    if attached_images:
        messages[-1].pop("images", None)

    autonomous_turns = 0
    tool_iterations = 0
    tool_failure_tracker = {}
    last_continue_payload = None
    seen_tool_requests = set()
    last_tool_result = ""
    cam_peek_saw_image = False
    peek_invoked = False

    watched_tags = _enabled_tags()

    # ====================== TOOL PROCESSING LOOP ======================
    while True:
        tool_iterations += 1
        runtime_context = build_runtime_context(
            autonomous_turns=autonomous_turns,
            tool_iterations=tool_iterations,
            autopilot_enabled=autopilot_enabled,
            autopilot_goal=autopilot_goal,
            autopilot_turns_used=autopilot_turns_used,
            memory_note=memory_note,
            mode=assistant_mode,
        )
        if tool_iterations > max_tool_iterations:
            response_text = (
                f"I've reached the tool-processing safety limit ({max_tool_iterations} iterations) for this turn. "
                "It appears I'm having trouble completing the task—please provide more guidance or break it into smaller steps."
            )
            log_event("tool:limit_reached", {"max": max_tool_iterations, "iterations": tool_iterations})
            break

        tagged_requests = []
        for tag in watched_tags:
            request = extract_tool_request(response_text, tag)
            if request:
                idx = response_text.find(f"<{tag}>")
                tagged_requests.append((idx, tag, request))

        if tagged_requests:
            tagged_requests.sort(key=lambda item: item[0])
            _, tag, (payload, _) = tagged_requests[0]

            log_event("tool:detected", {"tag": tag, "payload": payload})

            tool_key = (tag, payload[:100] if payload else "")

            if tool_key in seen_tool_requests:
                print(f"🚫 Repeated tool request detected for {tag}; stopping loop")
                _stream(f"🚫 Detected a repeated {tag} request — stopping to avoid a loop.")
                log_event("tool:repeat_detected", {"tag": tag, "payload": payload[:200]})
                if tag == "web_search" and get_last_search().get("query") == extract_search_mode(payload)[1]:
                    response_text = get_last_search().get("results", "") or "I already searched that, and I’m avoiding a loop."
                elif last_tool_result:
                    response_text = last_tool_result
                else:
                    response_text = "I detected a repeated tool request and stopped the loop."
                break

            seen_tool_requests.add(tool_key)

            if tag == "web_search":
                search_mode, search_query = extract_search_mode(payload)
                print(f"🌐 Jarvis tag-requested a web search for: {search_query}" + (f" (mode: {search_mode})" if search_mode else ""))
                _stream(f"🌐 Searching the web for: {search_query}")

                blocked_keywords = CONFIG.get("web_search", "blocked_keywords", default=[])
                blocked_hit = any(keyword.lower() in search_query.lower() for keyword in blocked_keywords)
                if len(search_query) < 25 and any(c.isdigit() for c in search_query):
                    print("🚫 Blocking unnecessary web search (numeric query)")
                    _stream("🚫 Skipped an unnecessary search.")
                    log_event("web_search:blocked", {"reason": "numeric_query", "query": search_query})
                    messages.append({"role": "assistant", "content": response_text})
                    break

                if blocked_hit:
                    print("🚫 Blocking web search due to restricted keywords")
                    _stream("🚫 Blocked a restricted search.")
                    log_event("web_search:blocked", {"reason": "restricted_keyword", "query": search_query})
                    messages.append({"role": "assistant", "content": response_text})
                    safe_reply = "I won't perform that search for safety reasons. Please ask something else."
                    messages.append({"role": "assistant", "content": safe_reply})
                    response_text = safe_reply
                    break

                search_results, was_cached = search_with_cache(search_query, mode=search_mode)
                if was_cached:
                    print("🔄 Re-using recent search results")
                    _stream(f"🔄 Reusing recent search results for: {search_query}")
                last_tool_result = search_results

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here are the web search results:\n\n{search_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "shell_exec":
                command = payload
                print(f"🛠️ Jarvis requested shell execution: {command}")
                _stream(f"🛠️ Running command: {command}")

                # Empty-command handling and the destructive-pattern blocklist are
                # now enforced inside perform_shell_exec itself (see shell.py), so
                # every caller gets the same protection rather than relying on
                # this one dispatch site to remember to check first.
                shell_results = perform_shell_exec(command)

                if "return code: 0" not in shell_results.lower() or "error" in shell_results.lower() or "cannot find" in shell_results.lower() or "not found" in shell_results.lower():
                    tool_failure_tracker[tool_key] = tool_failure_tracker.get(tool_key, 0) + 1
                    if tool_failure_tracker[tool_key] >= 2:
                        shell_results += (
                            "\n\n⚠️ WARNING: This command has failed multiple times with the same approach. "
                            "Consider trying a different method or asking the user for clarification."
                        )
                        log_event("tool:repeated_failure", {"tag": tag, "payload": payload[:200], "count": tool_failure_tracker[tool_key]})
                else:
                    tool_failure_tracker.pop(tool_key, None)
                last_tool_result = shell_results

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the result of the shell command:\n\n{shell_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "code_exec":
                print("💻 Jarvis requested code execution")
                _stream("💻 Executing an existing script...")
                code_results = perform_code_exec(payload)

                if "Error:" in code_results or "not found" in code_results.lower() or "return code: 0" not in code_results:
                    tool_failure_tracker[tool_key] = tool_failure_tracker.get(tool_key, 0) + 1
                    if tool_failure_tracker[tool_key] >= 2:
                        code_results += (
                            "\n\n⚠️ This execution attempt has failed repeatedly. "
                            "Verify the file exists and the payload format is correct."
                        )
                        log_event("tool:repeated_failure", {"tag": tag, "payload": payload[:200], "count": tool_failure_tracker[tool_key]})
                else:
                    tool_failure_tracker.pop(tool_key, None)
                last_tool_result = code_results

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the result of the code execution:\n\n{code_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "code_dev":
                print("🧩 Jarvis requested code development")
                _stream("🧩 Writing a code file...")
                code_results = perform_code_dev(payload)
                last_tool_result = code_results
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the result of the code development request:\n\n{code_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "get_environment":
                print("🔍 Jarvis requested environment context")
                _stream("🔍 Gathering environment context...")

                env_results = perform_get_environment()
                last_tool_result = env_results

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is your current environment context:\n\n{env_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "peek":
                print("👀 Jarvis requested a peek at the host desktop")
                _stream("👀 Peeking at what's running on the host desktop...")

                peek_results = perform_peek()
                last_tool_result = peek_results
                peek_invoked = True

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is what's currently running on the host desktop:\n\n{peek_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "cam_peek":
                if autopilot_enabled:
                    print("🚫 cam_peek blocked during autopilot mode")
                    _stream("🚫 Webcam peek is disabled during autopilot mode.")
                    log_event("cam_peek:blocked_autopilot")
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user",
                        "content": "cam_peek is not available during autopilot — it only runs when explicitly requested in a live turn."
                    })
                    response_text = generate_ollama_response(messages, runtime_context=runtime_context)
                else:
                    print("📷 Jarvis requested a webcam peek")
                    _stream("📷 Checking the webcam...")

                    cam_result = perform_cam_peek(telegram_chat_id=telegram_chat_id)
                    last_tool_result = cam_result["summary"]

                    messages.append({"role": "assistant", "content": response_text})
                    if cam_result["ok"] and cam_result.get("base64"):
                        cam_peek_saw_image = True
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Here is the webcam frame just captured. Detection summary: {cam_result['summary']}\n\n"
                                "Look at the image and respond to the user."
                            ),
                            "images": [cam_result["base64"]]
                        })
                        response_text = generate_ollama_response(messages, runtime_context=runtime_context)
                        messages[-1] = {
                            "role": "user",
                            "content": f"[Webcam frame captured and analyzed via cam_peek — {cam_result['summary']}]"
                        }
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"cam_peek failed: {cam_result['summary']}"
                        })
                        response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "transcribe_voice":
                voice_path = payload.strip()
                print(f"🎙️ Jarvis requested voice transcription: {voice_path}")
                _stream(f"🎙️ Re-transcribing voice file: {voice_path}")

                vosk_model_ref = runtime_state.get("vosk_model")
                if vosk_model_ref is None:
                    transcript_result = "Error: speech recognition model is not available in this session."
                else:
                    transcript_result = perform_transcribe_voice(voice_path, vosk_model_ref)
                last_tool_result = transcript_result

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"[Transcribed voice message]: {transcript_result}\n\n"
                        "Respond to this normally, in natural spoken language — your reply will be converted to a voice message and sent back."
                    )
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "ingest_image":
                image_path = payload.strip()
                print(f"🖼️ Jarvis requested image ingestion: {image_path}")
                _stream(f"🖼️ Loading image: {image_path}")

                image_result = perform_ingest_image(image_path)
                last_tool_result = "Image loaded." if image_result.get("ok") else image_result.get("error", "")

                messages.append({"role": "assistant", "content": response_text})
                if image_result.get("ok"):
                    messages.append({
                        "role": "user",
                        "content": f"Here is the image at {image_path}. Look at it and respond to the user's request about it.",
                        "images": [image_result["base64"]]
                    })
                    response_text = generate_ollama_response(messages, runtime_context=runtime_context)
                    messages[-1] = {
                        "role": "user",
                        "content": f"[Image previously analyzed by ingest_image: {image_path}]"
                    }
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Could not load the image: {image_result.get('error')}"
                    })
                    response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "read_file":
                file_path = payload.strip()
                print(f"📄 Jarvis requested file read: {file_path}")
                _stream(f"📄 Reading file: {file_path}")

                file_result = perform_read_file(file_path)
                last_tool_result = file_result

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the content of {file_path}:\n\n{file_result}"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "school_calendar":
                print(f"🗓️ Jarvis requested the school calendar (payload: {payload!r})")
                _stream("🗓️ Checking the school calendar...")

                calendar_results = perform_school_calendar(payload)
                last_tool_result = calendar_results

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the school calendar result:\n\n{calendar_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "gmail":
                print(f"📧 Jarvis requested Gmail (payload: {payload!r})")
                _stream("📧 Searching Gmail..." if payload else "📧 Checking recent email...")

                gmail_results = perform_gmail_check(payload)
                last_tool_result = gmail_results

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the Gmail result:\n\n{gmail_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "dj_play":
                dj_query = payload.strip()
                print(f"🎧 Jarvis requested DJ playback: {dj_query}")
                _stream(f"🎧 Starting playback: {dj_query}")

                dj_result = perform_dj_play(dj_query)
                last_tool_result = dj_result

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the result of the DJ play request:\n\n{dj_result}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "dj_stop":
                print("🎧 Jarvis requested DJ stop")
                _stream("🎧 Stopping playback...")

                dj_result = perform_dj_stop()
                last_tool_result = dj_result

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the result of the DJ stop request:\n\n{dj_result}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "dj_skip":
                print("🎧 Jarvis requested DJ skip")
                _stream("🎧 Skipping to the next track...")

                dj_result = perform_dj_skip()
                last_tool_result = dj_result

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the result of the DJ skip request:\n\n{dj_result}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "dj_queue_remove":
                dj_identifier = payload.strip()
                print(f"🎧 Jarvis requested DJ queue removal: {dj_identifier}")
                _stream(f"🎧 Removing from queue: {dj_identifier}")

                dj_result = perform_dj_queue_remove(dj_identifier)
                last_tool_result = dj_result

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the result of the DJ queue removal request:\n\n{dj_result}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "dj_queue_list":
                print("🎧 Jarvis requested DJ queue list")
                _stream("🎧 Checking the queue...")

                dj_result = perform_dj_queue_list()
                last_tool_result = dj_result

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the current DJ queue:\n\n{dj_result}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "soul_write":
                print(f"💭 Jarvis writing to soul: {payload[:60]}...")
                parts = payload.split("\n", 1)
                category = parts[0].strip() if len(parts) > 1 else "insight"
                entry = parts[1].strip() if len(parts) > 1 else parts[0].strip()

                soul_result = append_soul_entry(entry, category)
                last_tool_result = soul_result

                response_text = re.sub(r"<soul_write>.*?</soul_write>", "", response_text, flags=re.DOTALL).strip()
                if not response_text:
                    messages.append({"role": "assistant", "content": soul_result})
                    messages.append({"role": "user", "content": "Soul entry recorded. Continue with your response to the user."})
                    response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "continue":
                print(f"🔄 Jarvis requested continuation: {payload}")
                _stream(f"🔄 Still working on it{': ' + payload if payload else '...'}")
                log_event("autonomy:continue_requested", {"payload": payload})

                if payload == last_continue_payload:
                    print("🚫 Detected infinite continuation loop—breaking out")
                    log_event("autonomy:loop_detected", {"payload": payload})
                    response_text = "I seem to be stuck in a thinking loop. The task is complete."
                    break

                last_continue_payload = payload

                autonomous_turns += 1
                runtime_context = build_runtime_context(
                    autonomous_turns=autonomous_turns,
                    tool_iterations=tool_iterations,
                    autopilot_enabled=autopilot_enabled,
                    autopilot_goal=autopilot_goal,
                    autopilot_turns_used=autopilot_turns_used,
                    memory_note=memory_note,
                    mode=assistant_mode,
                )
                if autonomous_turns > max_autonomous_turns:
                    print(f"🚫 Autonomous limit reached ({max_autonomous_turns})")
                    log_event("autonomy:limit_reached", {"max": max_autonomous_turns})
                    response_text = "Reached thinking limit—please provide more input."
                    break

                continue_tag_pos = response_text.find("<continue>")
                thinking_before_continue = response_text[:continue_tag_pos].strip() if continue_tag_pos > 0 else ""

                if thinking_before_continue:
                    messages.append({"role": "assistant", "content": thinking_before_continue})

                continue_msg = payload if payload else "Continue."
                messages.append({"role": "user", "content": continue_msg})
                log_event("autonomy:continue_appended", continue_msg)

                new_response = generate_ollama_response(messages, runtime_context=runtime_context)
                if thinking_before_continue:
                    response_text = thinking_before_continue.strip() + "\n\n" + new_response
                else:
                    response_text = new_response
                log_event("llm:raw_response (autonomous)", response_text)
                continue

            continue
        else:
            break

    clean_response = sanitize_response(response_text)
    clean_response = guard_against_unverified_visual_claims(
        clean_response,
        cam_peek_saw_image=cam_peek_saw_image,
        peek_invoked=peek_invoked,
        had_attached_images=bool(attached_images),
    )
    messages.append({"role": "assistant", "content": clean_response})
    log_event("turn:end", {"response": clean_response})
    return clean_response
