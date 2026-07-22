"""DJ mode: yt-dlp -> ffplay pipeline plus a FIFO queue of upcoming tracks.

Position numbering (used in tool results and by dj_queue_remove) counts the
currently-playing track as position 1/N, so a single song shows as "1/1" and
the "queue" just ends when it finishes — songs added after it shift down as
2/N, 3/N...

Module-level + lock-protected because both tool calls (main thread) and the
background monitor thread (which auto-advances when a track finishes) read
and mutate this state.
"""
import difflib
import shutil
import subprocess
import threading
import time

from jarvis.capabilities import YTDLP_AVAILABLE, YTDLP_PATH
from jarvis.logging_utils import log_event
from jarvis.tools.shell import kill_process_tree

dj_lock = threading.Lock()
dj_state = {"ytdlp_proc": None, "ffplay_proc": None, "title": None, "current_query": None, "started_at": None}
dj_queue = []  # list of {"query": str, "title": str} dicts, in play order
dj_monitor_thread = None
dj_monitor_running = False


def _dj_probe_title(search_target: str) -> str:
    """Best-effort metadata-only lookup (no download) so we can tell the user
    what actually started playing. Not fatal if this fails or times out —
    playback can still proceed with the raw query as a fallback title."""
    try:
        result = subprocess.run(
            [YTDLP_PATH, "--print", "%(title)s", "--skip-download", "--quiet", "--no-warnings", search_target],
            capture_output=True, text=True, timeout=20
        )
        return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    except Exception as e:
        log_event("dj:probe_error", str(e))
        return ""


def _dj_kill_current_locked() -> bool:
    """Kill any currently-running yt-dlp/ffplay pair and clear dj_state, but
    leave dj_queue untouched. Caller must hold dj_lock. Returns True if
    something was actually stopped."""
    stopped_any = False
    for key in ("ffplay_proc", "ytdlp_proc"):
        proc = dj_state.get(key)
        if proc and proc.poll() is None:
            try:
                kill_process_tree(proc)
            except Exception as e:
                log_event("dj:stop_error", {"proc": key, "error": str(e)})
            stopped_any = True
        dj_state[key] = None
    dj_state["title"] = None
    dj_state["current_query"] = None
    dj_state["started_at"] = None
    return stopped_any


def _dj_launch_locked(track: dict) -> bool:
    """Start yt-dlp -> ffplay for a track dict {"query", "title"}. Assumes
    dj_lock is held and that any prior playback has already been cleared via
    _dj_kill_current_locked(). Returns True on success."""
    query = track["query"]
    title = track.get("title") or query
    ffplay_path = shutil.which("ffplay")
    if not YTDLP_AVAILABLE or not ffplay_path:
        log_event("dj:launch_skipped_missing_deps", {"query": query})
        return False

    search_target = query if query.lower().startswith(("http://", "https://")) else f"ytsearch1:{query}"
    try:
        ytdlp_proc = subprocess.Popen(
            [YTDLP_PATH, "-f", "bestaudio", "-o", "-", "--quiet", "--no-warnings", search_target],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        ffplay_proc = subprocess.Popen(
            [ffplay_path, "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
            stdin=ytdlp_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        ytdlp_proc.stdout.close()  # ffplay now owns the read end
    except Exception as e:
        log_event("dj:launch_error", str(e))
        return False

    dj_state["ytdlp_proc"] = ytdlp_proc
    dj_state["ffplay_proc"] = ffplay_proc
    dj_state["title"] = title
    dj_state["current_query"] = query
    dj_state["started_at"] = time.time()
    log_event("dj:playing", {"query": query, "title": title})
    print(f"🎧 DJ mode: now playing '{title}'")
    return True


def _dj_advance_locked():
    """Pop the next track off dj_queue and start it. Returns the started
    track dict, or None if the queue was empty / launch failed."""
    if not dj_queue:
        return None
    track = dj_queue.pop(0)
    return track if _dj_launch_locked(track) else None


def _dj_remove_current_locked():
    """Stop whatever's currently playing and auto-advance. Returns
    (removed_title_or_None, next_track_or_None)."""
    removed_title = dj_state.get("title")
    _dj_kill_current_locked()
    next_track = _dj_advance_locked()
    return removed_title, next_track


def _dj_format_queue_locked() -> str:
    """Render the combined 'now playing + upcoming' list as position/total
    lines. Assumes dj_lock is held."""
    total = (1 if dj_state.get("title") else 0) + len(dj_queue)
    if total == 0:
        return "Queue: empty."
    lines = [f"Queue ({total} total):"]
    idx = 1
    if dj_state.get("title"):
        lines.append(f"  {idx}/{total}: {dj_state['title']} (now playing)")
        idx += 1
    for track in dj_queue:
        lines.append(f"  {idx}/{total}: {track.get('title') or track['query']}")
        idx += 1
    return "\n".join(lines)


def _dj_monitor_loop():
    """Background thread target: polls the currently-playing ffplay process
    and, when it exits on its own, auto-advances to the next track in
    dj_queue."""
    global dj_monitor_running
    while dj_monitor_running:
        time.sleep(0.5)
        with dj_lock:
            proc = dj_state.get("ffplay_proc")
            if proc is not None and proc.poll() is not None:
                finished_title = dj_state.get("title")
                _dj_kill_current_locked()
                log_event("dj:track_finished", {"title": finished_title})
                _dj_advance_locked()
                # No TTS/Telegram announcement here by design — stays silent.


def start_dj_monitor():
    """Start the background auto-advance thread. Safe to call multiple times."""
    global dj_monitor_thread, dj_monitor_running
    if dj_monitor_thread and dj_monitor_thread.is_alive():
        return
    dj_monitor_running = True
    dj_monitor_thread = threading.Thread(target=_dj_monitor_loop, daemon=True)
    dj_monitor_thread.start()
    log_event("dj:monitor_started")


def stop_dj_monitor():
    """Stop the background auto-advance thread gracefully."""
    global dj_monitor_thread, dj_monitor_running
    dj_monitor_running = False
    if dj_monitor_thread and dj_monitor_thread.is_alive():
        dj_monitor_thread.join(timeout=2)
    dj_monitor_thread = None
    log_event("dj:monitor_stopped")


def dj_shutdown():
    """Called on program exit: stop the monitor thread and kill/clear everything."""
    stop_dj_monitor()
    with dj_lock:
        _dj_kill_current_locked()
        dj_queue.clear()


def _dj_probe_and_update_title_async(track: dict) -> None:
    """Background thread target: resolve the real title for `track` via
    yt-dlp (up to ~20s) and patch it into place once ready, without making
    the dj_play tool call that queued/started it wait around for it. `track`
    is the same dict instance that was appended to dj_queue (or used to
    launch playback), so mutating it here is visible everywhere else that
    still holds that reference (dj_queue_list, dj_queue_remove, etc).
    If the query was replaced/removed/skipped before this resolves, the
    guard below just makes this a harmless no-op."""
    query = track["query"]
    search_target = query if query.lower().startswith(("http://", "https://")) else f"ytsearch1:{query}"
    title = _dj_probe_title(search_target)
    if not title:
        return
    with dj_lock:
        track["title"] = title
        if dj_state.get("current_query") == query and dj_state.get("title") == query:
            dj_state["title"] = title
        log_event("dj:title_resolved", {"query": query, "title": title})


def perform_dj_play(query: str) -> str:
    """Tool: play a YouTube search query (or a direct URL) via yt-dlp, adding
    it to the DJ queue. Starts playback (or queues it) immediately using the
    raw query as a placeholder title — the real title is resolved in the
    background (see _dj_probe_and_update_title_async) so this call doesn't
    block the whole conversation turn on a ~20s yt-dlp metadata lookup before
    the user even hears anything start."""
    query = (query or "").strip()
    if not query:
        return "Error: no song, artist, or URL provided to play."
    if not YTDLP_AVAILABLE:
        return "Error: yt-dlp is not installed or not on PATH. Install it with: pip install yt-dlp"
    if not shutil.which("ffplay"):
        return "Error: ffplay not found on PATH. It ships alongside ffmpeg in full builds — check your ffmpeg install includes it."

    track = {"query": query, "title": query}

    with dj_lock:
        if dj_state.get("ffplay_proc") is None:
            ok = _dj_launch_locked(track)
            if not ok:
                return f"Error: could not start playback of '{query}'."
            summary = _dj_format_queue_locked()
            result = f"Now playing: {query} (resolving title...)\n{summary}"
        else:
            dj_queue.append(track)
            total = 1 + len(dj_queue)
            summary = _dj_format_queue_locked()
            result = f"Added to queue at position {total}/{total}: {query} (resolving title...)\n{summary}"

    threading.Thread(target=_dj_probe_and_update_title_async, args=(track,), daemon=True).start()
    return result


def perform_dj_skip() -> str:
    """Tool: stop whatever's currently playing and immediately advance to the
    next track in the queue (if any)."""
    with dj_lock:
        if not dj_state.get("title") and not dj_queue:
            return "Nothing is playing and the queue is empty — nothing to skip."
        removed_title, next_track = _dj_remove_current_locked()
        summary = _dj_format_queue_locked()
    log_event("dj:skipped", {"from": removed_title, "to": next_track.get("title") if next_track else None})
    next_title = (next_track.get("title") or next_track["query"]) if next_track else None
    if removed_title and next_title:
        return f"Skipped '{removed_title}'. Now playing: {next_title}\n{summary}"
    if next_title:
        return f"Now playing: {next_title}\n{summary}"
    if removed_title:
        return f"Skipped '{removed_title}'. Queue is now empty.\n{summary}"
    return f"Queue is now empty.\n{summary}"


def perform_dj_stop() -> str:
    """Tool: hard stop — kill whatever's playing AND clear the entire
    upcoming queue."""
    with dj_lock:
        was_playing = dj_state.get("title")
        queue_count = len(dj_queue)
        stopped = _dj_kill_current_locked()
        dj_queue.clear()
    log_event("dj:stopped", {"title": was_playing, "cleared_queue": queue_count})
    if stopped and was_playing and queue_count:
        return f"Stopped playback: {was_playing}. Cleared {queue_count} queued track(s)."
    if stopped and was_playing:
        return f"Stopped playback: {was_playing}."
    if queue_count:
        return f"Nothing was playing. Cleared {queue_count} queued track(s)."
    return "Nothing is currently playing and the queue is empty."


def perform_dj_queue_list() -> str:
    """Tool: report what's currently playing and what's queued up next."""
    with dj_lock:
        return _dj_format_queue_locked()


def perform_dj_queue_remove(identifier: str) -> str:
    """Tool: remove a track by its position number (1 = now playing, 2+ =
    upcoming queue in order) or by a fuzzy title/artist match. Removing
    position 1 stops the current track and auto-advances, same as dj_skip."""
    identifier = (identifier or "").strip()
    if not identifier:
        return "Error: no position number or song/artist text provided to remove."

    with dj_lock:
        total = (1 if dj_state.get("title") else 0) + len(dj_queue)
        if total == 0:
            return "Queue is empty — nothing to remove."

        target_key = None  # ("current",) or ("queue", idx)

        if identifier.isdigit():
            position = int(identifier)
            if position < 1 or position > total:
                summary = _dj_format_queue_locked()
                return f"Error: position {position} is out of range (queue has {total} track(s)).\n{summary}"
            if position == 1 and dj_state.get("title"):
                target_key = ("current",)
            else:
                queue_idx = (position - 2) if dj_state.get("title") else (position - 1)
                target_key = ("queue", queue_idx)
        else:
            combined = []
            if dj_state.get("title"):
                combined.append((("current",), dj_state["title"]))
            for i, track in enumerate(dj_queue):
                combined.append((("queue", i), track.get("title") or track["query"]))
            needle = identifier.lower()
            best_key, best_ratio = None, 0.0
            for key, title in combined:
                ratio = difflib.SequenceMatcher(None, needle, title.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio, best_key = ratio, key
            if best_key is None or best_ratio < 0.45:
                summary = _dj_format_queue_locked()
                log_event("dj:remove_no_match", {"identifier": identifier})
                return f"Error: no track matching '{identifier}' found in the queue.\n{summary}"
            target_key = best_key

        if target_key[0] == "current":
            removed_title, next_track = _dj_remove_current_locked()
            summary = _dj_format_queue_locked()
            log_event("dj:removed", {"identifier": identifier, "removed": removed_title, "was_current": True})
            if next_track:
                next_title = next_track.get("title") or next_track["query"]
                return f"Removed (was playing): {removed_title}. Now playing: {next_title}\n{summary}"
            return f"Removed (was playing): {removed_title}. Queue is now empty.\n{summary}"
        else:
            _, idx = target_key
            removed = dj_queue.pop(idx)
            summary = _dj_format_queue_locked()
            removed_title = removed.get("title") or removed["query"]
            log_event("dj:removed", {"identifier": identifier, "removed": removed_title, "was_current": False})
            return f"Removed from queue: {removed_title}\n{summary}"
