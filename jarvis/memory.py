"""Short-term (rolling conversation) memory: persistence, trimming, and
usage-pressure based condensation. Separate from soul.py, which handles the
private long-term journal."""
import json
import os
import re

from jarvis.config import CONFIG
from jarvis.logging_utils import log_event

DEFAULT_HISTORY_FILE = CONFIG.get("files", "conversation_history", default="conversation_history.json")


def load_conversation_history(filename: str = None):
    """Load .json conversation history, essentially short-term memory.
    This will always be generated to load system prompt."""
    filename = filename or DEFAULT_HISTORY_FILE
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


def save_conversation_history(messages, filename: str = None):
    """Persist messages to a .json file across sessions. Called after every turn."""
    filename = filename or DEFAULT_HISTORY_FILE
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        log_event("history:saved", {"file": filename, "messages": len(messages)})
    except Exception as e:
        log_event("history:save_error", {"file": filename, "error": str(e)})
        print(f"Warning: Could not save history: {e}")


def trim_history(messages, max_turns: int = None):
    """Keep system prompt + last N turns."""
    max_turns = max_turns if max_turns is not None else CONFIG.get("memory", "trim_max_turns", default=20)
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
    model_context = CONFIG.get("ollama", "model_context", default=131072)
    char_budget_multiplier = CONFIG.get("memory", "char_budget_multiplier", default=4)
    estimated_budget = max(1, model_context * char_budget_multiplier)
    estimated_used = len(serialized)
    estimated_ratio = min(1.0, estimated_used / estimated_budget)
    return estimated_ratio, estimated_used, estimated_budget


def build_memory_summary(messages, summary_char_limit: int = None) -> str:
    """Create a compact, condensed summary from older conversation turns by extracting key facts and decisions."""
    summary_char_limit = summary_char_limit if summary_char_limit is not None else CONFIG.get("memory", "summary_char_limit", default=1800)
    if not messages:
        return ""

    summary_parts = []
    last_user_intent = None
    assistant_outcomes = []
    key_facts = []

    for message in messages:
        role = message.get("role", "unknown")
        if role == "system":
            continue

        content = str(message.get("content", "")).strip()
        if not content or len(content) < 10:
            continue

        clean_content = re.sub(r"\s+", " ", content)

        if role == "user":
            if any(keyword in clean_content.lower() for keyword in ["want", "need", "help", "make", "create", "remember", "set", "prefer"]):
                last_user_intent = clean_content[:150]
        elif role == "assistant":
            if any(keyword in clean_content.lower() for keyword in ["done", "found", "set", "changed", "learned", "understood", "decided"]):
                assistant_outcomes.append(clean_content[:150])
            if any(keyword in clean_content.lower() for keyword in ["is ", "are ", "was ", "were ", "set to", "about "]):
                key_facts.append(clean_content[:150])

    if last_user_intent:
        summary_parts.append(f"User goal: {last_user_intent}")

    if key_facts:
        for fact in key_facts[:2]:
            summary_parts.append(f"Noted: {fact}")

    if assistant_outcomes:
        for outcome in assistant_outcomes[-2:]:
            summary_parts.append(f"Done: {outcome}")

    if not summary_parts:
        turn_count = len([m for m in messages if m.get("role") in ["user", "assistant"]])
        summary_parts.append(f"Earlier conversation ({turn_count} exchanges) - context preserved")

    summary_text = "[CONDENSED_MEMORY]\n" + "\n".join(summary_parts)

    if len(summary_text) > summary_char_limit:
        summary_text = summary_text[: summary_char_limit - 3].rstrip() + "..."

    return summary_text


def condense_conversation_history(messages, recent_turns: int = None) -> tuple:
    """Condense older conversation into a summary system message and keep recent turns."""
    recent_turns = recent_turns if recent_turns is not None else CONFIG.get("memory", "recent_turns_to_keep", default=8)
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
    check_interval = CONFIG.get("memory", "check_interval_prompts", default=3)
    warning_threshold = CONFIG.get("memory", "warning_threshold", default=0.50)
    condense_threshold = CONFIG.get("memory", "condense_threshold", default=0.75)

    state["memory_prompt_count"] = state.get("memory_prompt_count", 0) + 1
    prompt_count = state["memory_prompt_count"]

    if prompt_count % check_interval != 0:
        return messages, None, "", False

    usage_ratio, estimated_used, estimated_budget = estimate_memory_usage(messages, extra_text=incoming_text)
    memory_note = f"estimated memory use {usage_ratio * 100:.0f}% ({estimated_used}/{estimated_budget} chars)"
    log_event("memory:check", {"prompt_count": prompt_count, "usage_ratio": usage_ratio, "estimated_used": estimated_used, "estimated_budget": estimated_budget})

    warning_text = None
    if usage_ratio >= warning_threshold and not state.get("memory_warning_announced"):
        state["memory_warning_announced"] = True
        warning_text = f"Memory is getting crowded. I am at about {usage_ratio * 100:.0f}% of my short-term memory."
        log_event("memory:warning", {"prompt_count": prompt_count, "usage_ratio": usage_ratio})

    if usage_ratio >= condense_threshold:
        condensed_messages, did_condense, summary_text = condense_conversation_history(messages)
        if did_condense:
            messages[:] = condensed_messages
            save_conversation_history(messages, history_file)
            memory_note = f"{memory_note}; memory condensed to keep recent context"
            log_event("memory:condensed", {"prompt_count": prompt_count, "usage_ratio": usage_ratio, "summary_chars": len(summary_text), "messages": len(messages)})
            return messages, warning_text, memory_note, True

    return messages, warning_text, memory_note, False
