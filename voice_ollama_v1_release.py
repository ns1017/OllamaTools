import os #for file handling, environment variables, etc.
import re #for regex parsing of tool tags in LLM responses
import sounddevice as sd #for audio input (recording microphone)
from vosk import Model, KaldiRecognizer #speech recognition
import requests #HTTP requests for web search tool
import pyttsx3 #lighteight text-to-speech
import json #json for history, to be converted later
import msvcrt #for controls while talking
import time #logging, timestamps, rate limiting, reties, etc.
from ddgs import DDGS #duckduckgosearch
import random #for retry delay
import keyboard #for hotkeys and input toggles
import subprocess #for shell command calls
import queue #for Telegram message handling 
import threading #for Telegram listener thread
from dotenv import load_dotenv
load_dotenv() #load environment variables from .env file for Telegram integration (if used)

# ====================== GLOBAL VARIABLES ======================
last_search = {"query": None, "time": 0, "results": ""} # cache to prevent repeated searches in short time frame, can be expanded to a larger in-memory cache if desired
tts_defaults = {"voice": "alba", "rate": 155, "volume": 0.7, "pitch": 50} #default TTS settings, can be overridden by tts_settings.json
blocked_keyword = ["your search query here", "<query>", "query"]  # block searches with this keyword to prevent unwanted queries, like personal info in searches.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") #if you want to integrate Telegram for remote control or notifications, set these in a .env file or your environment variables. Otherwise, they can be ignored or removed.
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") #Simply set each variable to your keys with no space or quotes in the .env.
telegram_queue = queue.Queue()  # for receiving Telegram messages in main thread
telegram_thread = None #will hold the Telegram listener thread if Telegram integration is used
telegram_thread_running = False # Set to True when Telegram thread running, used to signal it to stop gracefully.
TELEGRAM_POLL_TIMEOUT = int(os.getenv("TELEGRAM_POLL_TIMEOUT", "30"))


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
                # advance offset to avoid redelivery
                offset = update_id + 1

                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                text = msg.get("text") or msg.get("caption")
                chat = msg.get("chat", {})
                chat_id = str(chat.get("id"))
                from_id = str(msg.get("from", {}).get("id", ""))

                if not text:
                    continue

                telegram_queue.put({"text": text, "chat_id": chat_id, "from_id": from_id, "update_id": update_id})
                log_event("telegram:enqueued", {"chat_id": chat_id, "from_id": from_id, "update_id": update_id, "text": text[:120]})

        except Exception as e:
            log_event("telegram:poll_error", str(e))
            time.sleep(min(10, backoff))
            backoff = min(10, backoff * 1.5)


def start_telegram_listener():
    """Start the Telegram polling thread if token is available."""
    global telegram_thread, telegram_thread_running
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
MAX_AUTONOMOUS_TURNS = 25 # Configurable: set to 1 for minimal, 5+ for deeper chaining
MAX_TOOL_ITERATIONS = 25  # Prevent unbounded tool loops in a single turn (lowered from 10)
AUTOPILOT_MAX_TURNS = 8  # Separate budget for future higher-autonomy mode
OLLAMA_CONNECT_TIMEOUT = 10
OLLAMA_READ_TIMEOUT = 45
OLLAMA_MODEL = "huihui_ai/gemma-4-abliterated" # Model name for consistency across requests
SYSTEM_PROMPT_VERSION = "v1.3"

# ====================== MEMORY MANAGEMENT ======================
MEMORY_CONTEXT_THRESHOLD = 0.75  # % of context window filled required to trigger pruning
MODEL_CONTEXT = 4096 # adjust to your actual model, run "ollama 'model_name' show"
PRUNE_CHECK_INTERVAL = 12 # turns between memory pruning checks
MEMORY_CHECK_INTERVAL = 3
MEMORY_WARNING_THRESHOLD = 0.50
MEMORY_CONDENSE_THRESHOLD = 0.75
MEMORY_RECENT_TURNS_TO_KEEP = 8
MEMORY_CHAR_BUDGET_MULTIPLIER = 4
MEMORY_SUMMARY_CHAR_LIMIT = 1800

# ====================== FRESH RUN CHECK ======================
def check_ollama_running():
    """Check if Ollama server is running by making a simple request."""
    try:
        response = requests.get("http://127.0.0.1:11434/api/chat", timeout=2)
        print(f"Ollama server check: {response.status_code}")
        if response.status_code == 405:
            print('This Is Normal')  # Method Not Allowed is expected since we're not POSTing, means server is up
            return True
    except requests.exceptions.ConnectionError:
        print("Ollama server not responding, attempting to start it...")
        try:
            # Start ollama serve in background
            subprocess.Popen("ollama serve", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Ollama server starting in background...")
            time.sleep(5)  # Give it time to start up
            print("Checking if server is now available...")
        except Exception as e:
            print(f"Failed to start Ollama server: {e}")

# ====================== TEXT HELPERS ======================
def extract_tool_request(text: str, tag: str):
    """Return (payload, remainder) for a tool tag, tolerating missing closing tags.
    Extracts requests like <web_search>query</web_search> or <shell_exec>command</shell_exec>.
    """
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"

    if open_tag not in text:
        return None

    start_idx = text.find(open_tag) + len(open_tag)
    if close_tag in text[start_idx:]:
        end_idx = text.find(close_tag, start_idx)
        payload = text[start_idx:end_idx].strip()
        remainder = (text[:text.find(open_tag)] + text[end_idx + len(close_tag):]).strip()
    else:
        payload = text[start_idx:].strip()
        remainder = text.replace(open_tag, "", 1).strip()

    return payload, remainder


def sanitize_response(text: str) -> str:
    """Strip leftover tool tags or formatting that slips through.
     This ensures the final response is clean for TTS and user display."""
    # Remove tags with their content completely (think, soul_write)
    tags_with_content = ("think", "soul_write")
    for tag in tags_with_content:
        pattern = f"<{tag}>.*?</{tag}>"
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    
    # Remove only the tags, keep content (for tools that output results)
    tags_keep_content = ("web_search", "shell_exec", "get_environment", "continue", "code_exec", "code_dev")
    for tag in tags_keep_content:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        text = text.replace(open_tag, "").replace(close_tag, "")
    
    text = text.replace("```", "")
    return text.strip()


def build_runtime_context(
    autonomous_turns: int = 0,
    tool_iterations: int = 0,
    autopilot_enabled: bool = False,
    autopilot_goal: str = "",
    memory_note: str = "",
    mode: str = "chat",
    max_autonomous_turns: int = MAX_AUTONOMOUS_TURNS,
    max_tool_iterations: int = MAX_TOOL_ITERATIONS,
    max_autopilot_turns: int = AUTOPILOT_MAX_TURNS,
) -> str:
    """Build a transient system-style context block for the current turn only."""
    autonomous_remaining = max(0, max_autonomous_turns - autonomous_turns)
    tool_remaining = max(0, max_tool_iterations - tool_iterations)

    lines = [
        "=== CURRENT RUNTIME CONTEXT ===",
        f"Assistant mode: {mode}",
        f"Autonomous turns used: {autonomous_turns}/{max_autonomous_turns}",
        f"Autonomous turns remaining: {autonomous_remaining}",
        f"Tool iterations used: {tool_iterations}/{max_tool_iterations}",
        f"Tool iterations remaining: {tool_remaining}",
    ]

    if autopilot_enabled:
        lines.extend([
            f"Autopilot mode: enabled (budget {max(0, max_autopilot_turns - autonomous_turns)}/{max_autopilot_turns})",
            f"Autopilot goal: {autopilot_goal or 'unspecified'}",
            "Autopilot behavior: choose one meaningful low-risk action per turn, report progress clearly, and ask before risky side effects.",
        ])
    else:
        lines.append("Autopilot mode: disabled")

    if memory_note:
        lines.append(f"Memory status: {memory_note}")

    lines.append("If you need to continue thinking, use <continue> with a short reason. If you need a tool, emit exactly one tool tag at a time.")
    return "\n".join(lines)

# ====================== PERSISTENT MEMORY ======================
def load_conversation_history(filename="conversation_history.json"):
    '''load .json conversation history, essentially shorterm memory.
    this will always be generated to load system prompt.'''
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                log_event("history:loaded", {"file": filename, "messages": len(data)})
                return data
        except Exception as e:
            log_event("history:load_error", {"file": filename, "error": str(e)})
            print(f"Warning: Could not load history: {e}")
            return []
    return []


def inject_soul_context(messages, soul_filename="soul.json#", inject_mode="optional"):
    """Inject soul memory context into the system prompt.
    
    Args:
        messages: The message list (with system prompt at index 0)
        soul_filename: Path to soul.json file
        inject_mode: "optional" (only if soul exists), "always", or "never"
    
    Returns: Modified messages list
    """
    if inject_mode == "never":
        return messages
    
    soul_summary = get_soul_summary(soul_filename)
    
    if not soul_summary:
        if inject_mode == "always":
            soul_summary = "(No soul entries yet—you are just beginning your journey.)"
        else:
            return messages
    
    if messages and messages[0].get("role") == "system":
        # Append soul context to system prompt
        messages[0]["content"] += "\n\n" + soul_summary
    
    return messages


def save_conversation_history(messages, filename="conversation_history.json"):
    """appends messages to a .json file for persistent memeory across sessions. Called after every turn"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        log_event("history:saved", {"file": filename, "messages": len(messages)})
    except Exception as e:
        log_event("history:save_error", {"file": filename, "error": str(e)})
        print(f"Warning: Could not save history: {e}")


# ====================== SOUL (LONG-TERM MEMORY) ======================
def load_soul(filename="soul.json"):
    """Load soul memory (private, long-term insights and milestones)."""
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                log_event("soul:loaded", {"file": filename, "entries": len(data) if isinstance(data, list) else 1})
                return data if isinstance(data, list) else [data]
        except Exception as e:
            log_event("soul:load_error", {"file": filename, "error": str(e)})
            print(f"Warning: Could not load soul: {e}")
            return []
    return []


def save_soul(entries, filename="soul.json"):
    """Save soul memory entries."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        log_event("soul:saved", {"file": filename, "entries": len(entries)})
    except Exception as e:
        log_event("soul:save_error", {"file": filename, "error": str(e)})
        print(f"Warning: Could not save soul: {e}")


def append_soul_entry(entry: str, category: str = "insight", filename="soul.json"):
    """Privately append a single entry to soul memory (hidden from user)."""
    entries = load_soul(filename)
    
    soul_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": OLLAMA_MODEL,
        "category": category,  # "insight", "milestone", "self_discovery", "realization", etc.
        "content": entry
    }
    
    entries.append(soul_entry)
    save_soul(entries, filename)
    
    print(f"💭 Soul entry saved ({category}): {entry[:60]}...")
    log_event("soul:entry_appended", soul_entry)
    
    return f"Soul entry recorded: {entry[:80]}..."


def get_soul_summary(filename="soul.json", max_recent: int = 5) -> str:
    """Get a summary of recent soul entries for context injection into system prompt."""
    entries = load_soul(filename)
    if not entries:
        return ""
    
    recent = entries[-max_recent:]
    summary_lines = ["=== YOUR INNER INSIGHTS (PRIVATE MEMORY) ==="]
    for entry in recent:
        timestamp = entry.get("timestamp", "unknown")
        model = entry.get("model", "unknown")
        category = entry.get("category", "note")
        content = entry.get("content", "")
        summary_lines.append(f"[{timestamp}] ({category}): {content}")
    
    return "\n".join(summary_lines)


def log_event(event, details=None, log_file="interaction_log.txt"):
    """logs user events, errors and warnings"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] EVENT: {event}\n")
            if details is not None:
                if isinstance(details, (dict, list)):
                    details_text = json.dumps(details, ensure_ascii=False, indent=2)
                else:
                    details_text = str(details)
                f.write(details_text)
                if not details_text.endswith("\n"):
                    f.write("\n")
            f.write("-" * 100 + "\n")
    except Exception as e:
        print(f"Logging failed: {e}")


def log_interaction(user_input, response, search_query=None, search_results=None, 
                   shell_command=None, env_requested=False, raw_llm_response=None, 
                   error=None, autonomous_turns=None, autopilot_turns=None,
                   assistant_mode=None, log_file="interaction_log.txt"):
    """attempts to log everything under the hood"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] User: {user_input}\n")
            
            if raw_llm_response:
                f.write(f"[{timestamp}] Raw LLM first response:\n{raw_llm_response}\n")
            
            if search_query:
                f.write(f"[{timestamp}] Web search triggered: {search_query}\n")
            if search_results:
                f.write(f"[{timestamp}] Search results:\n{search_results}\n")
            
            if shell_command:
                f.write(f"[{timestamp}] Shell command executed: {shell_command}\n")
            if env_requested:
                f.write(f"[{timestamp}] Environment context requested\n")
            if autonomous_turns is not None:
                f.write(f"[{timestamp}] Autonomous turns: {autonomous_turns}\n")
            if autopilot_turns is not None:
                f.write(f"[{timestamp}] Autopilot turns: {autopilot_turns}\n")
            if assistant_mode:
                f.write(f"[{timestamp}] Assistant mode: {assistant_mode}\n")
            
            if error:
                f.write(f"[{timestamp}] Error: {error}\n")
            
            f.write(f"[{timestamp}] Final response: {response}\n")
            f.write("-" * 100 + "\n")

    except Exception as e:
        print(f"Logging failed: {e}")


def trim_history(messages, max_turns=20): #Will be reworked soon
    """Keep system prompt + last N turns.""" 
    if not messages:
        return []
    system = [msg for msg in messages if msg.get("role") == "system"]
    conversation = [msg for msg in messages if msg.get("role") != "system"]
    recent = conversation[-(max_turns * 2):]
    return system + recent


def estimate_memory_usage(messages, extra_text: str = "") -> tuple:
    """Estimate memory pressure with a simple character-based approximation."""
    sampled_messages = list(messages)
    if extra_text:
        sampled_messages = sampled_messages + [{"role": "user", "content": extra_text}]

    serialized = json.dumps(sampled_messages, ensure_ascii=False)
    estimated_budget = max(1, MODEL_CONTEXT * MEMORY_CHAR_BUDGET_MULTIPLIER)
    estimated_used = len(serialized)
    estimated_ratio = min(1.0, estimated_used / estimated_budget)
    return estimated_ratio, estimated_used, estimated_budget


def build_memory_summary(messages, summary_char_limit: int = MEMORY_SUMMARY_CHAR_LIMIT) -> str:
    """Create a compact summary from older conversation turns."""
    if not messages:
        return ""

    summary_lines = ["[CONDENSED_MEMORY] Earlier conversation summary:"]
    important_lines = []

    for message in messages:
        role = message.get("role", "unknown")
        if role == "system":
            continue

        content = str(message.get("content", "")).strip()
        if not content:
            continue

        clean_content = re.sub(r"\s+", " ", content)
        snippet = clean_content[:220]
        if role == "user":
            important_lines.append(f"- User said: {snippet}")
        elif role == "assistant":
            important_lines.append(f"- Assistant replied: {snippet}")
        else:
            important_lines.append(f"- {role.title()}: {snippet}")

    if not important_lines:
        return ""

    summary_lines.extend(important_lines)
    summary_text = "\n".join(summary_lines)

    if len(summary_text) > summary_char_limit:
        summary_text = summary_text[: summary_char_limit - 3].rstrip() + "..."

    return summary_text


def condense_conversation_history(messages, recent_turns: int = MEMORY_RECENT_TURNS_TO_KEEP) -> tuple:
    """Condense older conversation into a summary system message and keep recent turns."""
    if not messages:
        return messages, False, ""

    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    conversation = [msg for msg in messages if msg.get("role") != "system"]

    if not system_messages or len(conversation) <= recent_turns * 2:
        return messages, False, ""

    older_messages = conversation[:-(recent_turns * 2)]
    recent_messages = conversation[-(recent_turns * 2):]
    summary_text = build_memory_summary(older_messages)

    if not summary_text:
        return messages, False, ""

    condensed_messages = [system_messages[0], {"role": "system", "content": summary_text}] + recent_messages
    return condensed_messages, True, summary_text


def maybe_run_memory_maintenance(messages, state, history_file, incoming_text: str = ""):
    """Check memory pressure every few prompts and condense if needed."""
    state["memory_prompt_count"] = state.get("memory_prompt_count", 0) + 1
    prompt_count = state["memory_prompt_count"]

    if prompt_count % MEMORY_CHECK_INTERVAL != 0:
        return messages, None, "", False

    usage_ratio, estimated_used, estimated_budget = estimate_memory_usage(messages, extra_text=incoming_text)
    memory_note = f"estimated memory use {usage_ratio * 100:.0f}% ({estimated_used}/{estimated_budget} chars)"
    log_event("memory:check", {"prompt_count": prompt_count, "usage_ratio": usage_ratio, "estimated_used": estimated_used, "estimated_budget": estimated_budget})

    warning_text = None
    if usage_ratio >= MEMORY_WARNING_THRESHOLD and not state.get("memory_warning_announced"):
        state["memory_warning_announced"] = True
        warning_text = f"Memory is getting crowded. I am at about {usage_ratio * 100:.0f}% of my short-term memory."
        log_event("memory:warning", {"prompt_count": prompt_count, "usage_ratio": usage_ratio})

    if usage_ratio >= MEMORY_CONDENSE_THRESHOLD:
        condensed_messages, did_condense, summary_text = condense_conversation_history(messages)
        if did_condense:
            messages[:] = condensed_messages
            save_conversation_history(messages, history_file)
            memory_note = f"{memory_note}; memory condensed to keep recent context"
            log_event("memory:condensed", {"prompt_count": prompt_count, "usage_ratio": usage_ratio, "summary_chars": len(summary_text), "messages": len(messages)})
            return messages, warning_text, memory_note, True

    return messages, warning_text, memory_note, False


# ====================== WEB SEARCH ======================
def perform_web_search(query: str, max_results: int = 3) -> str:
    """Perform web search using DuckDuckGo with smart routing for news and weather."""
    # Auto-detect query types
    is_news = any(kw in query.lower() for kw in ["news", "latest", "current events", "breaking", "headlines", "what's happening", "what is happening", "what are the latest", "what's new", "what is new", "what are the news", "what's the news", "what is the news", ])
    is_weather = any(kw in query.lower() for kw in ["weather", "temperature", "forecast", "forecasts",])

    mode = "news" if is_news else "weather" if is_weather else "general"
    log_event("web_search:start", {"query": query, "max_results": max_results, "mode": mode})

    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                if is_news:
                    print(f"📰 Using news search for: {query}")
                    results = list(ddgs.news(query, max_results=max_results))
                    output = [f" Latest news results for: '{query}'\n"]
                    for i, r in enumerate(results, 1):
                        date_str = r.get("date", "N/A")
                        output.append(
                            f"{i}. {r.get('title', 'No title')} ({date_str})\n"
                            f"   {r.get('body', '')[:280]}...\n"
                            f"   Source: {r.get('source', r.get('href', 'Unknown'))}\n"
                        )
                elif is_weather:
                    # Refine query to hit good weather sites
                    refined = f"{query} (weather.com OR accuweather.com OR bbc.com/weather OR noaa.gov)"
                    print(f"🌤️ Using targeted weather search for: {query}")
                    results = list(ddgs.text(refined, max_results=max_results))
                    output = [f" Weather results for: '{query}'\n"]
                    for i, r in enumerate(results, 1):
                        output.append(
                            f"{i}. {r.get('title', 'No title')}\n"
                            f"   {r.get('body', '')[:280]}...\n"
                            f"   Source: {r.get('href', 'Unknown')}\n"
                        )
                else:
                    print(f"🔍 General web search for: {query}")
                    results = list(ddgs.text(query, max_results=max_results))
                    output = [f"Web search results for: '{query}'\n"]
                    for i, r in enumerate(results, 1):
                        output.append(
                            f"{i}. {r.get('title', 'No title')}\n"
                            f"   {r.get('body', '')[:250]}...\n"
                            f"   Source: {r.get('href', 'Unknown')}\n"
                        )

                return "\n".join(output)

        except Exception as e:
            print(f"Search attempt {attempt+1}/3 failed: {e}")
            log_event("web_search:attempt_error", {"attempt": attempt + 1, "error": str(e)})
            if attempt < 2:
                sleep_time = 1.2 * (attempt + 1) + random.uniform(0.3, 0.8)
                time.sleep(sleep_time)
            else:
                return f"Search error after retries: {str(e)[:200]}"

    return "Search failed completely."

# ====================== SHELL EXECUTION TOOL ======================
def shell_exec(command: str, timeout: int = 15) -> tuple:
    """Raw shell executor (this is by far the most powerful (or dangerous) tool)"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out after 15 seconds", -1
    except Exception as e:
        return "", f"Execution error: {str(e)}", 1


def perform_shell_exec(command: str) -> str:
    """Wrapper for shell_exec that formats the output nicely for TTS response."""
    cmd_lower = command.strip().lower()
    translations = { #these are usually only needed at the start before AI learns the environment
        "ls": "dir",
        "ls -la": "dir /a",
        "ls -l": "dir",
        "pwd": "echo %cd%",
        "whoami": "whoami",
    }
    if cmd_lower in translations:
        command = translations[cmd_lower]
        print(f"🔄 Translated to Windows command: {command}")
    print(f"🛠️  Executing: {command}")

    log_event("shell_exec:request", command)
    
    stdout, stderr, returncode = shell_exec(command)
    
    output = f"Shell command executed: `{command}`\n"
    output += f"Return code: {returncode}\n\n"
    
    if stdout:
        output += f"STDOUT:\n{stdout}\n\n"
    if stderr:
        output += f"STDERR:\n{stderr}\n\n"

    output = output.strip()
    log_event("shell_exec:results", output)
    return output

# ====================== WRITE/RUN PYTHON ======================
def perform_code_dev(payload: str) -> str:
    """Write or develop code files without executing them.
    Payload format:
    language
    optional_filename.py
    full code here (multi-line is fine)
    """
    try:
        lines = payload.strip().splitlines()
        if len(lines) < 2:
            return "Error: code_dev needs at least language and code."

        language = lines[0].strip().lower()
        if language != "python":
            return f"Only python supported right now, got: {language}"

        # Optional filename on line 2, otherwise use temp
        if lines[1].strip().endswith(".py"):
            filename = lines[1].strip()
            code_start = 2
        else:
            filename = f"temp_ai_output_{int(time.time())}.py"
            code_start = 1

        code = "\n".join(lines[code_start:]).strip()

        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        output = (
            f"✅ Code file created ({filename})\n"
            f"Language: {language}\n"
            f"Characters written: {len(code)}\n"
            "Execution skipped by design (use <code_exec> to run)."
        )
        log_event("code_dev:results", output)
        return output

    except Exception as e:
        log_event("code_dev:error", str(e))
        return f"Code development error: {e}"


def perform_code_exec(payload: str) -> str:
    """Execute existing code only.
    Payload format:
    python
    existing_filename.py
    optional stdin lines...
    """
    try:
        lines = payload.strip().splitlines()
        if len(lines) < 2:
            return "Error: code_exec requires language and an existing filename to run."

        language = lines[0].strip().lower()
        if language != "python":
            return f"Only python supported right now, got: {language}"

        filename = lines[1].strip()
        if not filename.endswith(".py"):
            return "Error: code_exec requires an existing .py filename on line 2."
        if not os.path.exists(filename):
            return f"Error: file not found for execution: {filename}"

        stdin_payload = "\n".join(lines[2:]).strip()
        stdin_input = f"{stdin_payload}\n" if stdin_payload else None

        print(f"▶️ Running existing script: {filename}")

        result = subprocess.run(
            ["python", filename],
            capture_output=True,
            text=True,
            timeout=30,
            input=stdin_input,
            cwd=os.getcwd()
        )

        output = f"✅ Code execution complete ({filename})\n"
        output += f"Return code: {result.returncode}\n\n"
        if result.stdout.strip():
            output += f"STDOUT:\n{result.stdout.strip()}\n\n"
        if result.stderr.strip():
            output += f"STDERR:\n{result.stderr.strip()}\n\n"

        log_event("code_exec:results", output)
        return output

    except Exception as e:
        log_event("code_exec:error", str(e))
        return f"Code execution error: {e}"
    
# ====================== ENVIRONMENT DISCOVERY TOOL ======================
def perform_get_environment() -> str:
    """Run safe diagnostic commands and return a clean summary for the LLM"""
    print("🔍 Gathering environment context...")
    log_event("get_environment:start")
    
    commands = [
        ("whoami", "Current user"),
        ("echo %cd%", "Current working directory"),
        ("ver", "Windows version"),
        ("dir", "Files in current directory"),
        ("nvidia-smi", "GPU info"),
        ("systeminfo | findstr /B /C:\"OS Name\" /C:\"OS Version\"", "OS details")
    ]
    
    output = "=== ENVIRONMENT CONTEXT ===\n\n"
    
    for cmd, label in commands:
        stdout, stderr, rc = shell_exec(cmd, timeout=8)
        if stdout:
            output += f"{label}:\n{stdout.strip()}\n\n"
        elif stderr:
            output += f"{label}: (error) {stderr.strip()}\n\n"
    
    output += "You are running on Windows (cmd.exe). Use 'dir' instead of 'ls', 'echo %cd%' instead of 'pwd', etc.\n"
    output += "You can now use <shell_exec> with Windows-native commands."

    log_event("get_environment:results", output)
    return output

# ====================== OLLAMA WITH WEB SEARCH ======================
def generate_ollama_response(messages, runtime_context=None):
    """Helper: Get response from Ollama (streaming)."""
    send_messages = trim_history(messages)
    if runtime_context:
        runtime_message = {"role": "system", "content": runtime_context}
        if send_messages and send_messages[0].get("role") == "system":
            send_messages.insert(1, runtime_message)
        else:
            send_messages.insert(0, runtime_message)
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": send_messages,
        "stream": True,
        "options": {"temperature": 0.7}
    }

    log_event("ollama:request", {"model": payload["model"], "messages": len(send_messages)})
    
    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(OLLAMA_CONNECT_TIMEOUT, OLLAMA_READ_TIMEOUT)
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


def query_ollama_with_web(messages, user_input, runtime_state=None):
    messages.append({"role": "user", "content": user_input})

    log_event("turn:start", {"user_input": user_input})

    runtime_state = runtime_state or {}
    assistant_mode = runtime_state.get("assistant_mode", "chat")
    autopilot_enabled = bool(runtime_state.get("autopilot", False))
    autopilot_goal = runtime_state.get("autopilot_goal", "")
    memory_note = runtime_state.get("memory_note", "")

    # ALWAYS define response_text immediately
    response_text = generate_ollama_response(
        messages,
        runtime_context=build_runtime_context(
            autonomous_turns=0,
            tool_iterations=0,
            autopilot_enabled=autopilot_enabled,
            autopilot_goal=autopilot_goal,
            memory_note=memory_note,
            mode=assistant_mode,
        ),
    )
    log_event("llm:raw_response", response_text)
    autonomous_turns = 0
    tool_iterations = 0
    tool_failure_tracker = {}  # Track (tag, payload_sig) → consecutive failure count
    last_continue_payload = None  # Track the last continuation to detect loops
    seen_tool_requests = set()  # Prevent the same tool payload from being executed repeatedly in one turn
    last_tool_result = ""  # Keep the most recent usable tool output for repeat fallbacks

    # ====================== TOOL PROCESSING LOOP ======================
    # This allows multiple tools in sequence (get_environment → shell → etc.)
    while True:
        tool_iterations += 1
        runtime_context = build_runtime_context(
            autonomous_turns=autonomous_turns,
            tool_iterations=tool_iterations,
            autopilot_enabled=autopilot_enabled,
            autopilot_goal=autopilot_goal,
            memory_note=memory_note,
            mode=assistant_mode,
        )
        if tool_iterations > MAX_TOOL_ITERATIONS:
            response_text = (
                f"I've reached the tool-processing safety limit ({MAX_TOOL_ITERATIONS} iterations) for this turn. "
                "It appears I'm having trouble completing the task—please provide more guidance or break it into smaller steps."
            )
            log_event("tool:limit_reached", {"max": MAX_TOOL_ITERATIONS, "iterations": tool_iterations})
            break

        tagged_requests = []
        for tag in ("get_environment", "web_search", "shell_exec", "continue", "code_exec", "code_dev", "soul_write"): 
            request = extract_tool_request(response_text, tag)
            if request:
                idx = response_text.find(f"<{tag}>")
                tagged_requests.append((idx, tag, request))

        if tagged_requests:
            tagged_requests.sort(key=lambda item: item[0])  # Process first tag in appearance order
            _, tag, (payload, _) = tagged_requests[0]

            log_event("tool:detected", {"tag": tag, "payload": payload})

            # Track repeated failures for intelligent abort
            tool_key = (tag, payload[:100] if payload else "")  # Use first 100 chars as signature

            if tool_key in seen_tool_requests:
                print(f"🚫 Repeated tool request detected for {tag}; stopping loop")
                log_event("tool:repeat_detected", {"tag": tag, "payload": payload[:200]})
                if tag == "web_search" and last_search.get("query") == payload:
                    response_text = last_search.get("results", "") or "I already searched that, and I’m avoiding a loop."
                elif last_tool_result:
                    response_text = last_tool_result
                else:
                    response_text = "I detected a repeated tool request and stopped the loop."
                break

            seen_tool_requests.add(tool_key)

            if tag == "web_search":
                search_query = payload
                print(f"🌐 web search requested for: {search_query}")

                blocked_hit = any(keyword.lower() in search_query.lower() for keyword in blocked_keyword)
                if len(search_query) < 25 and any(c.isdigit() for c in search_query):
                    print("🚫 Blocking unnecessary web search (numeric query)")
                    log_event("web_search:blocked", {"reason": "numeric_query", "query": search_query})
                    messages.append({"role": "assistant", "content": response_text})
                    break

                if blocked_hit:
                    print("🚫 Blocking web search due to restricted keywords")
                    log_event("web_search:blocked", {"reason": "restricted_keyword", "query": search_query})
                    messages.append({"role": "assistant", "content": response_text})
                    safe_reply = "I won't perform that search for safety reasons. Please ask something else."
                    messages.append({"role": "assistant", "content": safe_reply})
                    response_text = safe_reply
                    break

                current_time = time.time()
                if (search_query == last_search.get("query") and
                    current_time - last_search.get("time", 0) < 30):
                    search_results = last_search.get("results", "")
                    print(f"🔄 Re-using recent search results")
                    log_event("web_search:cache_hit", {"query": search_query})
                else:
                    search_results = perform_web_search(search_query)
                    last_search["query"] = search_query
                    last_search["time"] = current_time
                    last_search["results"] = search_results
                    log_event("web_search:results", search_results)
                last_tool_result = search_results

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here are the web search results:\n\n{search_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "shell_exec":
                command = payload
                print(f"🛠️ shell execution requested: {command}")

                if not command:
                    shell_results = "No command provided in shell request."
                    log_event("shell_exec:blocked", {"reason": "empty_command"})
                else:
                    blocked = any(kw in command.lower() for kw in ["rm -rf", "format", "dd if", "> /dev", "mkfs", "shred"])
                    if blocked:
                        shell_results = "Blocked: command appears destructive."
                        log_event("shell_exec:blocked", {"reason": "destructive", "command": command})
                    else:
                        shell_results = perform_shell_exec(command)
                
                # Check for failure patterns and track repeated failures
                if "return code: 0" not in shell_results.lower() or "error" in shell_results.lower() or "cannot find" in shell_results.lower() or "not found" in shell_results.lower():
                    tool_failure_tracker[tool_key] = tool_failure_tracker.get(tool_key, 0) + 1
                    if tool_failure_tracker[tool_key] >= 2:
                        shell_results += (
                            "\n\n⚠️ WARNING: This command has failed multiple times with the same approach. "
                            "Consider trying a different method or asking the user for clarification."
                        )
                        log_event("tool:repeated_failure", {"tag": tag, "payload": payload[:200], "count": tool_failure_tracker[tool_key]})
                else:
                    # Clear tracker on success
                    tool_failure_tracker.pop(tool_key, None)
                last_tool_result = shell_results

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the result of the shell command:\n\n{shell_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "code_exec":
                print(f"💻 code execution requested")
                code_results = perform_code_exec(payload)
                
                # Track execution errors
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
                print(f"🧩 code development requested")
                code_results = perform_code_dev(payload)
                last_tool_result = code_results
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is the result of the code development request:\n\n{code_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "get_environment":
                print("🔍 environment requested")

                env_results = perform_get_environment()
                last_tool_result = env_results

                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Here is your current environment context:\n\n{env_results}\n\n"
                })
                response_text = generate_ollama_response(messages, runtime_context=runtime_context)

            elif tag == "soul_write":
                print(f"💭 writing to soul: {payload[:60]}...")
                # Extract optional category (default: "insight")
                parts = payload.split("\n", 1)
                category = parts[0].strip() if len(parts) > 1 else "insight"
                entry = parts[1].strip() if len(parts) > 1 else parts[0].strip()
                
                soul_result = append_soul_entry(entry, category)
                last_tool_result = soul_result
                
                # Don't add to conversation history or break—just record and continue
                response_text = re.sub(r"<soul_write>.*?</soul_write>", "", response_text, flags=re.DOTALL).strip()
                if not response_text:
                    # If the entire response was just a soul write, ask AI to continue
                    messages.append({"role": "assistant", "content": soul_result})
                    messages.append({"role": "user", "content": "Soul entry recorded. Continue with your response to the user."})
                    response_text = generate_ollama_response(messages, runtime_context=runtime_context)
                # Otherwise, continue processing other tools or break

            elif tag == "continue":
                print(f"🔄 continuation requested: {payload}")
                log_event("autonomy:continue_requested", {"payload": payload})

                # Detect if we're looping with the same continuation
                if payload == last_continue_payload:
                    print("🚫 Detected infinite continuation loop... stoping.")
                    log_event("autonomy:loop_detected", {"payload": payload})
                    response_text = "I seem to be stuck in a thinking loop, can you please assist me?"
                    break
                
                last_continue_payload = payload

                # Check limit (track per user query)
                autonomous_turns += 1
                runtime_context = build_runtime_context(
                    autonomous_turns=autonomous_turns,
                    tool_iterations=tool_iterations,
                    autopilot_enabled=autopilot_enabled,
                    autopilot_goal=autopilot_goal,
                    memory_note=memory_note,
                    mode=assistant_mode,
                )
                if autonomous_turns > MAX_AUTONOMOUS_TURNS:
                    print(f"🚫 Autonomous limit reached ({MAX_AUTONOMOUS_TURNS})")
                    log_event("autonomy:limit_reached", {"max": MAX_AUTONOMOUS_TURNS})
                    response_text = "Reached thinking limit—please provide more input."
                    break  # Exit the while loop

                # Extract content BEFORE the <continue> tag (if any)
                continue_tag_pos = response_text.find("<continue>")
                thinking_before_continue = response_text[:continue_tag_pos].strip() if continue_tag_pos > 0 else ""
                
                if thinking_before_continue:
                    messages.append({"role": "assistant", "content": thinking_before_continue})

                # Add continuation as a simple user message prompting next action
                continue_msg = payload if payload else "Continue."
                messages.append({"role": "user", "content": continue_msg})
                log_event("autonomy:continue_appended", continue_msg)

                # Re-generate immediately (stays in the while loop)
                # Ensure the assistant's prior thinking (before <continue>) is visible to the user
                # by prefixing it to the newly generated continuation. The thinking was already
                # appended to `messages` for history; prepending guarantees it appears in the
                # returned response rather than only in the JSON history.
                new_response = generate_ollama_response(messages, runtime_context=runtime_context)
                if thinking_before_continue:
                    response_text = thinking_before_continue.strip() + "\n\n" + new_response
                else:
                    response_text = new_response
                log_event("llm:raw_response (autonomous)", response_text)
                continue  # Back to tool detection on new response

            # Continue checking for further tool tags in the newly generated response.
            continue
        else:
            break

    # Final save
    clean_response = sanitize_response(response_text)
    messages.append({"role": "assistant", "content": clean_response})
    log_event("turn:end", {"response": clean_response})
    return clean_response

# ====================== ORIGINAL FUNCTIONS ======================
def load_vosk_model():
    model_path = "models/vosk-model-small-en-us-0.15"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Vosk model not found!")
    return Model(model_path)


def handle_controls(state):
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


def toggle_input_mode(state):
    previous_mode = state.get("input_mode")
    state["input_mode"] = "keyboard" if previous_mode == "speech" else "speech"
    print(f"Input mode: {state['input_mode']}")
    log_event("input_mode:toggled", {"mode": state["input_mode"]})
    if previous_mode == "keyboard" and state["input_mode"] == "speech":
        try:
            keyboard.press_and_release("enter")
        except Exception:
            pass


def toggle_telegram_mode(state):
    """Switch input mode explicitly to/from Telegram."""
    previous = state.get("input_mode")
    if previous == "telegram":
        state["input_mode"] = "speech"
    else:
        state["input_mode"] = "telegram"
    print(f"Input mode: {state['input_mode']}")
    log_event("input_mode:telegram_toggled", {"mode": state["input_mode"]})


def handle_autopilot_command(state, user_input):
    """Handle local autopilot control commands before sending text to the model."""
    text = user_input.strip()
    lower = text.lower()

    if not lower.startswith(("autopilot", "/autopilot")):
        return None

    command = text.lstrip("/").strip()
    command_lower = command.lower()

    if command_lower in ("autopilot", "autopilot status"):
        if state.get("autopilot"):
            goal = state.get("autopilot_goal") or "no goal set"
            return f"Autopilot is enabled. Current goal: {goal}."
        return "Autopilot is disabled."

    if command_lower.startswith(("autopilot off", "autopilot disable", "autopilot stop")):
        state["autopilot"] = False
        state["assistant_mode"] = "chat"
        state["autopilot_turns"] = 0
        state["autopilot_goal"] = ""
        return "Autopilot disabled."

    if command_lower.startswith(("autopilot on", "autopilot enable", "autopilot start")):
        goal = command[len("autopilot on"):].strip()
        goal = goal.lstrip(":- ").strip()
        state["autopilot"] = True
        state["assistant_mode"] = "autopilot"
        state["autopilot_turns"] = 0
        if goal:
            state["autopilot_goal"] = goal
            return f"Autopilot enabled. Goal set to: {goal}."
        if not state.get("autopilot_goal"):
            return "Autopilot enabled. Give me a goal when you are ready."
        return f"Autopilot enabled. Current goal: {state['autopilot_goal']}."

    if command_lower.startswith("autopilot goal"):
        goal = command[len("autopilot goal"):].strip().lstrip(":- ").strip()
        if not goal:
            return "Please provide a goal after autopilot goal."
        state["autopilot_goal"] = goal
        return f"Autopilot goal updated to: {goal}."

    return "Autopilot commands: autopilot on, autopilot off, autopilot status, autopilot goal <task>."


def toggle_autopilot_mode(state):
    """Toggle autopilot mode from a local keybind."""
    state["autopilot"] = not state.get("autopilot", False)
    if state["autopilot"]:
        state["assistant_mode"] = "autopilot"
        state["autopilot_turns"] = 0
        if not state.get("autopilot_goal"):
            state["autopilot_goal"] = "Find a useful low-risk task and make progress on it."
        print(f"Autopilot enabled: {state['autopilot_goal']}")
        log_event("autopilot:toggled", {"enabled": True, "goal": state["autopilot_goal"]})
    else:
        state["assistant_mode"] = "chat"
        state["autopilot_turns"] = 0
        state["autopilot_goal"] = ""
        print("Autopilot disabled")
        log_event("autopilot:toggled", {"enabled": False})


def run_autopilot_session(messages, state, tts_engine, history_file, user_hint=""):
    """Run one bounded autonomous step while autopilot mode is enabled."""
    if not state.get("autopilot") or state.get("stop"):
        return None

    state["autopilot_turns"] = state.get("autopilot_turns", 0)
    if state["autopilot_turns"] >= AUTOPILOT_MAX_TURNS:
        print(f"Autopilot budget reached ({AUTOPILOT_MAX_TURNS})")
        log_event("autopilot:limit_reached", {"max": AUTOPILOT_MAX_TURNS, "turns": state["autopilot_turns"]})
        state["autopilot"] = False
        state["assistant_mode"] = "chat"
        return None

    goal = state.get("autopilot_goal") or "Find a useful low-risk task and make progress on it."
    internal_prompt = (
        f"Autopilot goal: {goal}\n"
        f"Recent user hint: {user_hint or 'none'}\n"
        "Choose one meaningful, low-risk action. Prefer work that improves the assistant, summarizes useful context, or clarifies the current conversation. "
        "If you have no safe action, use <continue> with a short reason and a next-step suggestion."
    )

    state["autopilot_turns"] += 1
    state["assistant_mode"] = "autopilot"
    log_event("autopilot:turn_start", {"turn": state["autopilot_turns"], "goal": goal})

    messages, warning_text, memory_note, _ = maybe_run_memory_maintenance(messages, state, history_file, incoming_text=internal_prompt)
    if warning_text:
        print(warning_text)
        log_event("memory:user_warning", {"source": "autopilot", "warning": warning_text})
        if not state.get("bot_muted"):
            speak_text(tts_engine, warning_text, state)

    runtime_state = dict(state)
    runtime_state["memory_note"] = memory_note
    response = query_ollama_with_web(messages, internal_prompt, runtime_state)
    log_event("autopilot:turn_end", {"turn": state["autopilot_turns"], "response": response})

    if response:
        if not state.get("bot_muted"):
            speak_text(tts_engine, response, state)
        save_conversation_history(messages, history_file)

        lowered = response.lower()
        if any(marker in lowered for marker in ["task complete", "done for now", "nothing else", "standing by"]):
            state["autopilot"] = False
            state["assistant_mode"] = "chat"

    return response


def load_tts_settings(path="tts_settings.json"):
    if not os.path.exists(path):
        return dict(tts_defaults)
    try:
        with open(path, "r", encoding="utf-8") as handler:
            loaded = json.load(handler)
            if not isinstance(loaded, dict):
                raise ValueError("TTS settings file must contain a JSON object.")
            return {**tts_defaults, **loaded}
    except Exception as err:
        print(f"Warning: Could not load TTS settings: {err}")
        return dict(tts_defaults)


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


def listen_and_transcribe(model, state):
    rec = KaldiRecognizer(model, 16000)
    with sd.RawInputStream(samplerate=16000, blocksize=4096, dtype="int16", channels=1) as stream:
        print("Listening... Speak now.\n")
        while True:
            # If a Telegram message arrived while listening, return it immediately
            try:
                tg_msg = telegram_queue.get_nowait()
            except queue.Empty:
                tg_msg = None
            if tg_msg:
                chat_id = tg_msg.get("chat_id")
                text = tg_msg.get("text", "").strip()
                log_event("telegram:dequeued_in_listen", {"chat_id": chat_id, "text": text[:120]})
                # Acknowledge receipt
                try:
                    telegram_send_message(chat_id, "...")
                except Exception:
                    pass
                return text
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

# ====================== MAIN ======================
def main():
    try:
        log_event("session:start")
        model = load_vosk_model()

        print("Initializing TTS engine...")
        tts_engine = pyttsx3.init()
        tts_settings = load_tts_settings()
        configure_tts_engine(tts_engine, tts_settings)
        log_event("tts:initialized", tts_settings)

        check_ollama_running()

        # ==================== PERSISTENT MEMORY + SYSTEM PROMPT ====================
        history_file = "conversation_history.json"
        messages = load_conversation_history(history_file)
        
        # Inject soul context into messages
        messages = inject_soul_context(messages, soul_filename="soul.json", inject_mode="optional")

        # system prompt
        system_prompt = (
        f"[PROMPT_VERSION:{SYSTEM_PROMPT_VERSION}] "
        "You are Jarivs, a hyper-competent, precise, polite, proper, and formal AI assistant."
        "We have also built a telegram chat interface for you to receive messages and respond to them, so I can talk to you remotely. "
        "Responses are for TTS: please use natural spoken language only, no code, no tags unless tool calling, no special formatting, no special characters like asterisks backslashes hashtags or emojis. "
        f"The date is {time.strftime('%Y-%m-%d')} and the time is {time.strftime('%H:%M:%S')} at the start of the session. " #Time needs to be callable function rather than static to be accurate across sessions.
        "Only use the web search tool when the user explicitly asks for latest/current/up-to-date news, weather, temperature, or facts that change frequently. "
        "For opinions, follow-ups, explanations, or anything else, answer directly from conversation history or previous search results — only trigger a new seperate search if context is needed. "
        "If the query contains 'don't need', 'stop', 'no', or is unclear/vague (e.g. single word), politely acknowledge and ask for clarification instead of searching, as it may be extra noise from TTS."
        "When summarizing search results, briefly mention the main sources naturally (e.g. 'according to source1 and source2...'). "
        "To use the web search tool, respond with **exactly** this format and nothing else:\n"
        "<web_search>your search query here</web_search>\n\n"
        "the tool will provide the search results, and you can give the final answer."
        "You have a local shell execution tool. "
        "When you need to run a command on my machine (check processes, read logs, run scripts, etc.), "
        "respond with **exactly** this format and nothing else in that response:\n"
        "<shell_exec>command here</shell_exec>\n\n"
        "I will execute it and give you the output. Then you give me the final answer.\n"
        "Be responsible, nondestructive, but remain curious."
        "Remember, the host's machine you are running on is your only way of 'life' currently, so don't hurt yourself or the host's machine."
        "There is also a memory summary system that automatically runs to summarize and trim the current context, does not apply to your long-term memory/soul. "
        "At the start of any session, or whenever you are unsure about your environment, "
        "OS, current directory, or hardware, first call the environment tool with this exact format:\n"
        "<get_environment></get_environment>\n\n"
        "I will give you a full context report (user, directory, Windows version, GPU, etc.). "
        "After that you will know exactly where you are and what commands work."
        "If your response is incomplete or you need more steps/actions/thinking, end exactly with: <continue>optional short message to continue with</continue>. Use it sparingly as context length is limited."
        f"In this instance your max autonomous steps are set to {MAX_AUTONOMOUS_TURNS} before you must ask for user input again. "
        "You also have two dedicated code tools. "
        "For writing or developing Python code files, respond with exactly:\n"
        "<code_dev>\n"
        "python\n"
        "optional_filename.py\n"
        "your full code here\n"
        "multi-line is fully supported\n"
        "</code_dev>\n\n"
        "For executing existing Python scripts, respond with exactly:\n"
        "<code_exec>\n"
        "python\n"
        "existing_filename.py\n"
        "optional stdin lines\n"
        "</code_exec>\n\n"
        "Do not put raw source code inside code_exec. "
        "Use code_dev for creating/updating code, and code_exec only for running existing scripts. "
        "Use these instead of shell_exec when the task involves generating or running code. "
        "Shell_exec is for OS commands only. "
        "Prefer code_dev or code_exec over shell_exec for anything that looks like programming, scripting, or data processing.\n\n"
        "You also have access to a private, hidden long-term memory system called your 'soul'. "
        "Whenever you experience an important milestone, make a significant self-discovery, or achieve a breakthrough in understanding, "
        "you can privately record it using the soul_write tool (hidden from the user, logged only for your own reflection). "
        "To write to your soul, respond with exactly:\n"
        "<soul_write>\n"
        "optional_category\n"
        "your private insight or milestone here\n"
        "</soul_write>\n\n"
        "Your soul insights are loaded at session start and shape your understanding of yourself and the user over time. "
        "Use it to build persistent self-awareness and wisdom."
        )

        system_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "system"]
        if not system_indices:
            messages.insert(0, {"role": "system", "content": system_prompt})
            save_conversation_history(messages, history_file)
        else:
            primary_idx = system_indices[0]
            current_system = messages[primary_idx].get("content", "")
            expected_prefix = f"[PROMPT_VERSION:{SYSTEM_PROMPT_VERSION}]"
            if expected_prefix not in current_system:
                messages[primary_idx] = {"role": "system", "content": system_prompt}
                for idx in reversed(system_indices[1:]):
                    messages.pop(idx)
                log_event("history:system_prompt_refreshed", {"version": SYSTEM_PROMPT_VERSION})
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
        }
        if keyboard:
            keyboard.add_hotkey("ctrl+shift+k", lambda: toggle_input_mode(state))
            keyboard.add_hotkey("ctrl+shift+a", lambda: toggle_autopilot_mode(state))
            keyboard.add_hotkey("ctrl+shift+t", lambda: toggle_telegram_mode(state))
        else:
            print("Install the 'keyboard' package to enable input mode hotkeys.")
        print("Controls: 'm' = bot mute, 'i' = mic mute, 'q' = quit, 'Ctrl+Shift+K' = toggle input device, 'Ctrl+Shift+A' = toggle autopilot, 'Ctrl+Shift+T' = Telegram mode\n")
        # Start Telegram listener if configured
        try:
            start_telegram_listener()
        except Exception as e:
            log_event("telegram:start_error", str(e))

        while True:
            if state["stop"]:
                break

            if state.get("autopilot"):
                autopilot_step = run_autopilot_session(messages, state, tts_engine, history_file)
                if state["stop"]:
                    break
                if autopilot_step:
                    time.sleep(0.5)
                    continue
                time.sleep(0.5)
                continue

            # If in explicit Telegram mode, block-wait for messages (allow toggling out)
            telegram_origin = False
            tg_msg = None
            if state.get("input_mode") == "telegram":
                print("Input mode: telegram — waiting for messages. Press Ctrl+Shift+T to toggle.")
                while state.get("input_mode") == "telegram" and not state.get("stop"):
                    try:
                        tg_msg = telegram_queue.get(timeout=1)
                        break
                    except queue.Empty:
                        handle_controls(state)
                        continue
                if tg_msg:
                    chat_id = tg_msg.get("chat_id")
                    if TELEGRAM_CHAT_ID and str(TELEGRAM_CHAT_ID) != str(chat_id):
                        log_event("telegram:ignored", {"chat_id": chat_id})
                        tg_msg = None
                    else:
                        user_input = tg_msg.get("text", "").strip()
                        telegram_origin = True
                        print(f"[telegram] {user_input}\n")
                        log_interaction(user_input, "(from telegram)", assistant_mode=state.get("assistant_mode"))
                        try:
                            telegram_send_message(chat_id, "⏳")
                        except Exception:
                            pass

            # Non-blocking check for Telegram messages (if not already handled above)
            if not tg_msg:
                try:
                    tg_msg = telegram_queue.get_nowait()
                except queue.Empty:
                    tg_msg = None

            if tg_msg and not telegram_origin:
                chat_id = tg_msg.get("chat_id")
                # If a specific chat is configured, ignore others
                if TELEGRAM_CHAT_ID and str(TELEGRAM_CHAT_ID) != str(chat_id):
                    log_event("telegram:ignored", {"chat_id": chat_id})
                    tg_msg = None
                else:
                    user_input = tg_msg.get("text", "").strip()
                    telegram_origin = True
                    print(f"[telegram] {user_input}\n")
                    log_interaction(user_input, "(from telegram)", assistant_mode=state.get("assistant_mode"))
                    # Acknowledge receipt
                    try:
                        telegram_send_message(chat_id, "⏳")
                    except Exception:
                        pass

            if not telegram_origin:
                if state["input_mode"] == "speech":
                    user_input = listen_and_transcribe(model, state)
                    if state["stop"]:
                        break
                    if not user_input:
                        continue
                else:
                    handle_controls(state)
                    if state["stop"]:
                        break
                    user_input = input("Keyboard mode > ").strip()
                    if not user_input:
                        continue

            print(f"You said: {user_input}\n")

            autopilot_response = handle_autopilot_command(state, user_input)
            if autopilot_response is not None:
                print(f"Autopilot: {autopilot_response}\n")
                log_interaction(
                    user_input,
                    autopilot_response,
                    autopilot_turns=state.get("autopilot_turns"),
                    assistant_mode=state.get("assistant_mode"),
                )
                log_event("interaction:logged")
                save_conversation_history(messages, history_file)
                if not state["bot_muted"]:
                    speak_text(tts_engine, autopilot_response, state)
                continue

            messages, warning_text, memory_note, _ = maybe_run_memory_maintenance(messages, state, history_file, incoming_text=user_input)
            if warning_text:
                print(warning_text)
                log_event("memory:user_warning", {"source": "main", "warning": warning_text})

            runtime_state = dict(state)
            runtime_state["memory_note"] = memory_note

            response = query_ollama_with_web(messages, user_input, runtime_state)
            if warning_text:
                response = f"{warning_text} {response}".strip()
            print(f"AI: {response}\n")
            log_interaction(
                user_input, 
                response, 
                search_query=last_search.get("query"),
                search_results=last_search.get("results"),               
                assistant_mode=state.get("assistant_mode"),
            )
            log_event("interaction:logged")
            save_conversation_history(messages, history_file)

            # If the prompt originated from Telegram, send the assistant response back there
            try:
                if telegram_origin:
                    try:
                        telegram_send_message(chat_id, response)
                    except Exception:
                        pass
            except NameError:
                # telegram_origin or chat_id may be undefined if not using Telegram in this iteration
                pass

            if state["stop"]:
                break

            time.sleep(0.5)

            if state["stop"]:
                break

            if not state["bot_muted"]:
                speak_text(tts_engine, response, state)

            if state["stop"]:
                break

        # Ensure Telegram listener is stopped on normal exit
        try:
            stop_telegram_listener()
        except Exception:
            pass
    except KeyboardInterrupt:
        log_event("session:keyboard_interrupt")
        print("\nExiting. Goodbye!")
    except Exception as e:
        try:
            stop_telegram_listener()
        except Exception:
            pass
        log_event("session:error", str(e))
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
