"""Plain-text event/interaction logging (interaction_log.txt by default,
path configurable via config.json -> files.interaction_log)."""
import json
import time

from jarvis.config import CONFIG

DEFAULT_LOG_FILE = CONFIG.get("files", "interaction_log", default="interaction_log.txt")


def log_event(event, details=None, log_file: str = None):
    """Logs user events, errors and warnings."""
    log_file = log_file or DEFAULT_LOG_FILE
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
                     assistant_mode=None, log_file: str = None):
    """Attempts to log everything under the hood for a single turn."""
    log_file = log_file or DEFAULT_LOG_FILE
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

            f.write(f"[{timestamp}] Final Jarvis response: {response}\n")
            f.write("-" * 100 + "\n")

    except Exception as e:
        print(f"Logging failed: {e}")
