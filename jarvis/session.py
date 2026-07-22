"""Session-level state control: input-mode toggles (speech/keyboard/telegram),
autopilot on/off + one bounded autonomous step, and the /reset command."""
import time

import keyboard

from jarvis.audio_io import speak_text
from jarvis.config import CONFIG
from jarvis.logging_utils import log_event
from jarvis.memory import maybe_run_memory_maintenance, save_conversation_history
from jarvis.ollama_client import query_ollama_with_web
from jarvis.soul import inject_soul_context


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


def _build_fresh_system_message(system_prompt: str) -> dict:
    """Build the system message exactly as it's assembled at startup: base
    system_prompt plus the current soul summary, if any. Used by /reset so a
    reset always reflects up-to-date long-term memory rather than a stale
    snapshot taken back when the session first started."""
    temp = [{"role": "system", "content": system_prompt}]
    temp = inject_soul_context(temp, inject_mode="optional")
    return temp[0]


def handle_reset_command(state, user_input, messages, history_file, system_prompt):
    """Detect and handle a /reset (clear conversation history) command from
    either host or Telegram input. Gated behind a short confirmation step
    since this is destructive and easy to trigger by accident — a mishearing
    over voice, or a stray Telegram message. Soul (long-term memory) is left
    untouched; this only clears the rolling conversation.
    Returns a text reply to send back, or None if user_input wasn't this command."""
    reset_confirm_window = CONFIG.get("turns", "reset_confirm_window_seconds", default=120)
    text = user_input.strip()
    lower = text.lower().lstrip("/").strip()

    if lower in ("reset", "reset history", "reset context", "reset memory", "clear history", "clear context"):
        state["pending_reset_confirm"] = time.time()
        log_event("history:reset_requested")
        return (
            "This will permanently clear our conversation history — long-term memory (soul) is untouched. "
            f"Send 'reset confirm' within {reset_confirm_window} seconds to proceed, or ignore this to cancel."
        )

    if lower in ("reset confirm", "confirm reset"):
        pending_at = state.get("pending_reset_confirm")
        state["pending_reset_confirm"] = None
        if not pending_at or (time.time() - pending_at) > reset_confirm_window:
            return "No pending reset to confirm (or it expired) — send 'reset' first."

        fresh_system_message = _build_fresh_system_message(system_prompt)
        messages.clear()
        messages.append(fresh_system_message)
        save_conversation_history(messages, history_file)
        log_event("history:reset_confirmed")
        return "Done — conversation history has been reset. I'm starting fresh from here; long-term memory is untouched."

    return None


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
    autopilot_max_turns = CONFIG.get("turns", "autopilot_max_turns", default=8)

    if not state.get("autopilot") or state.get("stop"):
        return None

    state["autopilot_turns"] = state.get("autopilot_turns", 0)
    if state["autopilot_turns"] >= autopilot_max_turns:
        print(f"Autopilot budget reached ({autopilot_max_turns})")
        log_event("autopilot:limit_reached", {"max": autopilot_max_turns, "turns": state["autopilot_turns"]})
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
    runtime_state["vosk_model"] = state.get("vosk_model")
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
