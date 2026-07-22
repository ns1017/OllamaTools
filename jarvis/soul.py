"""Soul memory: a private, long-term journal of insights/milestones the
assistant records about itself, separate from the rolling conversation
history. File path configurable via config.json -> files.soul."""
import json
import os
import time

from jarvis.config import CONFIG
from jarvis.logging_utils import log_event

DEFAULT_SOUL_FILE = CONFIG.get("files", "soul", default="soul.json")


def load_soul(filename: str = None):
    """Load soul memory (private, long-term insights and milestones)."""
    filename = filename or DEFAULT_SOUL_FILE
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


def save_soul(entries, filename: str = None):
    """Save soul memory entries."""
    filename = filename or DEFAULT_SOUL_FILE
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        log_event("soul:saved", {"file": filename, "entries": len(entries)})
    except Exception as e:
        log_event("soul:save_error", {"file": filename, "error": str(e)})
        print(f"Warning: Could not save soul: {e}")


def append_soul_entry(entry: str, category: str = "insight", filename: str = None):
    """Privately append a single entry to soul memory (hidden from user)."""
    filename = filename or DEFAULT_SOUL_FILE
    entries = load_soul(filename)

    soul_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": CONFIG.get("ollama", "model"),
        "category": category,  # "insight", "milestone", "self_discovery", "realization", etc.
        "content": entry,
    }

    entries.append(soul_entry)
    save_soul(entries, filename)

    print(f"💭 Soul entry saved ({category}): {entry[:60]}...")
    log_event("soul:entry_appended", soul_entry)

    return f"Soul entry recorded: {entry[:80]}..."


def get_soul_summary(filename: str = None, max_recent: int = 5) -> str:
    """Get a summary of recent soul entries for context injection into system prompt."""
    filename = filename or DEFAULT_SOUL_FILE
    entries = load_soul(filename)
    if not entries:
        return ""

    recent = entries[-max_recent:]
    summary_lines = ["=== YOUR INNER INSIGHTS (PRIVATE MEMORY) ==="]
    for entry in recent:
        timestamp = entry.get("timestamp", "unknown")
        category = entry.get("category", "note")
        content = entry.get("content", "")
        summary_lines.append(f"[{timestamp}] ({category}): {content}")

    return "\n".join(summary_lines)


def inject_soul_context(messages, soul_filename: str = None, inject_mode: str = "optional"):
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
        messages[0]["content"] += "\n\n" + soul_summary

    return messages
