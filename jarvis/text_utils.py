"""Text-level helpers used by the Ollama tool-calling loop: extracting tool
tags from model output, cleaning the final response for TTS/display, and a
couple of heuristic guards against local-model confabulation."""
import difflib
import re
import time

from jarvis.config import CONFIG
from jarvis.logging_utils import log_event

# Every tool tag the assistant might emit. Used to strip stray/incomplete
# tags out of the final response even when a tool is feature-disabled.
ALL_TOOL_TAGS = (
    "web_search", "shell_exec", "get_environment", "continue", "code_exec", "code_dev",
    "transcribe_voice", "ingest_image", "read_file", "dj_play", "dj_stop", "dj_skip",
    "dj_queue_remove", "dj_queue_list", "peek", "cam_peek", "school_calendar", "gmail",
)


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
    tags_with_content = ("think", "soul_write")
    for tag in tags_with_content:
        pattern = f"<{tag}>.*?</{tag}>"
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    for tag in ALL_TOOL_TAGS:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        text = text.replace(open_tag, "").replace(close_tag, "")

    text = text.replace("```", "")
    return text.strip()


_SEARCH_MODE_PREFIX_RE = re.compile(r"^\s*\[mode:\s*(news|weather|general)\s*\]\s*", re.IGNORECASE)


def extract_search_mode(payload: str):
    """Pull an optional leading '[mode:news|weather|general]' prefix off a
    web_search payload, e.g. '[mode:news] latest AI safety research'.
    Returns (mode_or_None, remaining_query). If the model already knows what
    kind of search it wants, this skips perform_web_search's keyword-based
    guessing (which routes on words like 'weather' or 'latest' appearing
    anywhere in the query) and uses the explicit choice instead."""
    payload = payload or ""
    match = _SEARCH_MODE_PREFIX_RE.match(payload)
    if not match:
        return None, payload.strip()
    return match.group(1).lower(), payload[match.end():].strip()


# Known misspellings/mishearings of "peek" worth catching outright (voice transcription via
# Vosk is a common source of these), plus a fuzzy fallback via difflib for anything else close.
_PEEK_TYPO_WORDS = {"peak", "peaked", "peaks", "peep", "peeped", "peeps"}


def detect_peek_typo_hint(text: str):
    """Heuristic check for near-miss phrasing of the peek/cam_peek trigger words (e.g. a
    mistyped or mis-transcribed 'cam peak' instead of 'cam peek'). Returns a short hint
    string to append to this turn if a near-miss is detected, else None.

    This does NOT force a tool call — it only gives the model a stronger nudge than the raw
    (possibly garbled) text alone, since local models tend to confabulate an answer on
    ambiguous input rather than asking for clarification or calling the tool anyway."""
    if not text:
        return None
    lowered = text.lower()
    words = re.findall(r"[a-z]+", lowered)
    if not words:
        return None

    peek_like = any(w in _PEEK_TYPO_WORDS for w in words) or any(
        len(w) >= 3 and w != "peek" and difflib.SequenceMatcher(None, w, "peek").ratio() >= 0.7
        for w in words
    )
    if not peek_like:
        return None

    has_cam_context = any(w in ("cam", "camera", "webcam") for w in words)
    if has_cam_context:
        return (
            "Note: the phrasing above is close to, but not exactly, a webcam-check request. "
            "If the user is asking to check the webcam, use the cam_peek tool now rather than "
            "guessing or describing an image you have not actually captured."
        )
    return (
        "Note: the phrasing above is close to, but not exactly, a desktop-activity check. "
        "If the user is asking what's running on their screen, use the peek tool now rather than guessing."
    )


# Heuristic phrase lists for the post-hoc grounding check below. Tune these if you see
# false positives (legitimate replies getting blocked) or false negatives (confabulation
# slipping through with different wording).
CAM_CLAIM_PHRASES = (
    "i can see you", "the webcam shows", "webcam shows", "looking at the webcam",
    "i'm looking at you", "im looking at you", "through the webcam", "the camera shows",
    "on the webcam right now", "i can see in the room", "i see in the room",
)
DESKTOP_CLAIM_PHRASES = (
    "you currently have open", "you have open right now", "running on your screen",
    "on your desktop right now", "your screen shows", "currently running on your desktop",
    "i can see you're running", "i can see you are running", "what you have open right now",
)


def guard_against_unverified_visual_claims(text: str, cam_peek_saw_image: bool, peek_invoked: bool,
                                            had_attached_images: bool = False) -> str:
    """Defense-in-depth against a local model confabulating a cam_peek/peek result instead of
    calling the tool or asking for clarification on ambiguous/mistyped input. If the final
    response makes a live-webcam or live-desktop claim but the corresponding tool never
    actually ran this turn (and no image was otherwise attached), the model is almost
    certainly making it up — replace the response with an honest one instead of letting the
    confabulation reach the user."""
    lowered = text.lower()

    if not cam_peek_saw_image and not had_attached_images:
        if any(phrase in lowered for phrase in CAM_CLAIM_PHRASES):
            log_event("cam_peek:unverified_claim_blocked", {"response": text})
            return (
                "I haven't actually checked the webcam this turn, so I don't want to guess or make "
                "that up — say 'cam peek' (or 'check the webcam') and I'll take an actual look."
            )

    if not peek_invoked:
        if any(phrase in lowered for phrase in DESKTOP_CLAIM_PHRASES):
            log_event("peek:unverified_claim_blocked", {"response": text})
            return (
                "I haven't actually looked at your desktop this turn, so I don't want to guess — "
                "say 'peek' and I'll take an actual snapshot of what's running."
            )

    return text


def build_runtime_context(
    autonomous_turns: int = 0,
    tool_iterations: int = 0,
    autopilot_enabled: bool = False,
    autopilot_goal: str = "",
    autopilot_turns_used: int = 0,
    memory_note: str = "",
    mode: str = "chat",
    max_autonomous_turns: int = None,
    max_tool_iterations: int = None,
    max_autopilot_turns: int = None,
) -> str:
    """Build a transient system-style context block for the current turn only."""
    max_autonomous_turns = max_autonomous_turns if max_autonomous_turns is not None else CONFIG.get("turns", "max_autonomous_turns", default=12)
    max_tool_iterations = max_tool_iterations if max_tool_iterations is not None else CONFIG.get("turns", "max_tool_iterations", default=12)
    max_autopilot_turns = max_autopilot_turns if max_autopilot_turns is not None else CONFIG.get("turns", "autopilot_max_turns", default=8)

    autonomous_remaining = max(0, max_autonomous_turns - autonomous_turns)
    tool_remaining = max(0, max_tool_iterations - tool_iterations)

    lines = [
        "=== CURRENT RUNTIME CONTEXT ===",
        f"Current date: {time.strftime('%Y-%m-%d')}",
        f"Current time: {time.strftime('%H:%M:%S')}",
        f"Assistant mode: {mode}",
        f"Autonomous turns used: {autonomous_turns}/{max_autonomous_turns}",
        f"Autonomous turns remaining: {autonomous_remaining}",
        f"Tool iterations used: {tool_iterations}/{max_tool_iterations}",
        f"Tool iterations remaining: {tool_remaining}",
    ]

    if autopilot_enabled:
        autopilot_remaining = max(0, max_autopilot_turns - autopilot_turns_used)
        lines.extend([
            f"Autopilot mode: enabled (turn {autopilot_turns_used}/{max_autopilot_turns}, {autopilot_remaining} remaining)",
            f"Autopilot goal: {autopilot_goal or 'unspecified'}",
            "Autopilot behavior: choose one meaningful low-risk action per turn, report progress clearly, and ask before risky side effects.",
        ])
    else:
        lines.append("Autopilot mode: disabled")

    if memory_note:
        lines.append(f"Memory status: {memory_note}")

    lines.append("If you need to continue thinking, use <continue> with a short reason. If you need a tool, emit exactly one tool tag at a time.")
    return "\n".join(lines)
