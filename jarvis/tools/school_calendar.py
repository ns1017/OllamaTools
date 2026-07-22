"""School calendar tool: periodically fetches a public, read-only .ics feed
(no auth required) and lets the model check upcoming events on demand.
Recurring events (weekly classes, etc.) are expanded via
recurring_ical_events so a weekly-recurring block shows up correctly for
whatever date range is asked for, not just its original DTSTART.

Cached to disk so a network hiccup doesn't leave Jarvis with nothing to
say, and so repeated calls in a short window don't re-fetch every time —
see config.json -> school_calendar.cache_ttl_seconds.
"""
import datetime
import os
import time

import icalendar
import recurring_ical_events
import requests

from jarvis.config import CONFIG
from jarvis.logging_utils import log_event

ICS_URL = CONFIG.get("school_calendar", "ics_url", default="")
CACHE_FILE = CONFIG.get("school_calendar", "cache_file", default="school_calendar_cache.ics")
CACHE_TTL_SECONDS = CONFIG.get("school_calendar", "cache_ttl_seconds", default=1800)
DEFAULT_LOOKAHEAD_DAYS = CONFIG.get("school_calendar", "default_lookahead_days", default=7)
MAX_EVENTS_RETURNED = CONFIG.get("school_calendar", "max_events_returned", default=20)

# In-memory cache for this process; the on-disk copy (CACHE_FILE) is the
# fallback if a live fetch fails or the URL isn't reachable at all.
_cache = {"raw_ics": None, "fetched_at": 0}


def _load_disk_cache() -> str:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                raw = f.read()
            log_event("school_calendar:used_disk_cache", {"bytes": len(raw)})
            return raw
        except Exception as e:
            log_event("school_calendar:disk_cache_read_error", str(e))
    return ""


def _fetch_ics(force: bool = False) -> str:
    """Return raw .ics text: fresh in-memory cache if within TTL, otherwise a
    live fetch, falling back to the on-disk cache if the fetch fails or no
    URL is configured."""
    now = time.time()
    if not force and _cache["raw_ics"] and (now - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
        return _cache["raw_ics"]

    if not ICS_URL:
        log_event("school_calendar:no_url_configured")
        return _load_disk_cache()

    try:
        resp = requests.get(ICS_URL, timeout=15)
        resp.raise_for_status()
        raw = resp.text
        _cache["raw_ics"] = raw
        _cache["fetched_at"] = now
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(raw)
        except Exception as e:
            log_event("school_calendar:disk_cache_write_error", str(e))
        log_event("school_calendar:fetched", {"bytes": len(raw)})
        return raw
    except Exception as e:
        log_event("school_calendar:fetch_error", str(e))
        return _load_disk_cache()


def _normalize_start(dt):
    """Return a naive datetime usable as a sort key, whether the source
    value was a date (all-day event), a naive datetime, or a tz-aware
    datetime. Returns None if dt isn't a date/datetime at all."""
    if isinstance(dt, datetime.datetime):
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    if isinstance(dt, datetime.date):
        return datetime.datetime.combine(dt, datetime.time.min)
    return None


def _format_event(event) -> str:
    start = event.get("DTSTART").dt if event.get("DTSTART") else None
    end = event.get("DTEND").dt if event.get("DTEND") else None
    summary = str(event.get("SUMMARY", "Untitled event"))
    location = event.get("LOCATION")

    def fmt(dt):
        if isinstance(dt, datetime.datetime):
            return dt.strftime("%a %b %d, %I:%M %p")
        if isinstance(dt, datetime.date):
            return dt.strftime("%a %b %d (all day)")
        return str(dt)

    line = f"- {fmt(start)}"
    if (
        isinstance(start, datetime.datetime)
        and isinstance(end, datetime.datetime)
        and end.date() == start.date()
    ):
        line += f" - {end.strftime('%I:%M %p')}"
    line += f": {summary}"
    if location:
        line += f" @ {location}"
    return line


def get_upcoming_events(days: int = None) -> list:
    """Return a list of (sort_key_datetime, formatted_line) tuples for
    events starting within the next `days` days, expanded for recurrence
    and sorted chronologically. sort_key_datetime may be None for a
    malformed event with no usable start; those sort last."""
    days = days if days is not None else DEFAULT_LOOKAHEAD_DAYS
    raw = _fetch_ics()
    if not raw:
        return []

    try:
        calendar = icalendar.Calendar.from_ical(raw)
    except Exception as e:
        log_event("school_calendar:parse_error", str(e))
        return []

    now = datetime.datetime.now()
    end_window = now + datetime.timedelta(days=days)

    try:
        occurrences = recurring_ical_events.of(calendar).between(now, end_window)
    except Exception as e:
        log_event("school_calendar:expand_error", str(e))
        return []

    results = []
    for event in occurrences:
        start_raw = event.get("DTSTART").dt if event.get("DTSTART") else None
        sort_key = _normalize_start(start_raw)
        results.append((sort_key, _format_event(event)))

    results.sort(key=lambda pair: (pair[0] is None, pair[0] or datetime.datetime.min))
    return results


def perform_school_calendar(payload: str = "") -> str:
    """Tool entry point: return a formatted list of upcoming school calendar
    events. Payload is an optional number of days to look ahead (defaults
    to config -> school_calendar.default_lookahead_days, clamped 1-60);
    anything non-numeric is ignored and the default is used."""
    payload = (payload or "").strip()
    days = DEFAULT_LOOKAHEAD_DAYS
    if payload:
        try:
            days = max(1, min(60, int(payload)))
        except ValueError:
            pass

    if not ICS_URL:
        return "Error: no school calendar URL configured (set school_calendar.ics_url in config.json)."

    events = get_upcoming_events(days)
    if not events:
        return f"No school calendar events found in the next {days} day(s) (or the feed could not be read)."

    events = events[:MAX_EVENTS_RETURNED]
    lines = [f"School calendar — next {days} day(s):"]
    lines.extend(line for _, line in events)
    if len(events) == MAX_EVENTS_RETURNED:
        lines.append(f"(truncated to {MAX_EVENTS_RETURNED} events)")
    return "\n".join(lines)
