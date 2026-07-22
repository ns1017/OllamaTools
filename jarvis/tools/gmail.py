"""Gmail tool: read-only inbox viewing and search via the Gmail API.

Lists recent messages or searches using Gmail's own query syntax (from:,
subject:, is:unread, newer_than:7d, etc.). Uses the gmail.readonly OAuth
scope only — there is no send/delete/modify path anywhere in this module,
so even a compromised prompt can't use it to do anything but look.

First-time setup (one time only):
  1. In Google Cloud Console, enable the Gmail API and create an OAuth
     client ID of type "Desktop app".
  2. Download the client secret JSON and save it as the file named in
     config.json -> gmail.credentials_file (default: gmail_credentials.json)
     in the project root.
  3. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
  4. The first time the gmail tool actually runs, a browser window will
     open asking you to sign in and grant read-only access. After that,
     the resulting token is cached to config.json -> gmail.token_file
     (default: gmail_token.json) and refreshed automatically — no further
     browser prompts.

gmail_credentials.json and gmail_token.json are both secrets (the token
file grants live read access to the inbox) and are blocked from the
read_file tool via config.json -> security.file_access_blocked_names,
the same way .env is.
"""
import os
import time
from email.utils import parsedate_to_datetime

from jarvis.capabilities import GMAIL_LIBS_AVAILABLE
from jarvis.config import CONFIG
from jarvis.logging_utils import log_event

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CREDENTIALS_FILE = CONFIG.get("gmail", "credentials_file", default="gmail_credentials.json")
TOKEN_FILE = CONFIG.get("gmail", "token_file", default="gmail_token.json")
DEFAULT_MAX_RESULTS = CONFIG.get("gmail", "max_results", default=10)
CACHE_TTL_SECONDS = CONFIG.get("gmail", "cache_ttl_seconds", default=60)

# Cached authenticated client for this process (auth is slow; the API call itself is not).
_service = None
# Short-lived result cache, same idea as web_search's, so the model asking
# the same thing twice in a row doesn't re-hit the API needlessly.
_cache = {"key": None, "time": 0, "result": ""}


def _get_credentials():
    """Load cached OAuth credentials, refreshing silently if expired, or
    running the one-time browser consent flow if no valid token exists yet.
    Returns a Credentials object, or None if auth isn't possible right now."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            log_event("gmail:token_load_error", str(e))
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            log_event("gmail:token_refreshed")
            return creds
        except Exception as e:
            log_event("gmail:token_refresh_error", str(e))
            creds = None

    if not os.path.exists(CREDENTIALS_FILE):
        log_event("gmail:no_credentials_file", {"expected": CREDENTIALS_FILE})
        return None

    try:
        print(f"📧 Gmail: no valid token found — opening a browser window for one-time authorization "
              f"(using {CREDENTIALS_FILE})...")
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        print(f"📧 Gmail: authorized and token saved to {TOKEN_FILE}.")
        log_event("gmail:authorized_new_token")
        return creds
    except Exception as e:
        log_event("gmail:oauth_flow_error", str(e))
        print(f"❌ Gmail authorization failed: {e}")
        return None


def _get_service():
    """Lazily build (and cache) an authenticated Gmail API client for this
    process. Returns None if libraries aren't installed or auth fails."""
    global _service
    if _service is not None:
        return _service
    if not GMAIL_LIBS_AVAILABLE:
        return None

    from googleapiclient.discovery import build

    creds = _get_credentials()
    if not creds:
        return None
    try:
        _service = build("gmail", "v1", credentials=creds)
        return _service
    except Exception as e:
        log_event("gmail:build_service_error", str(e))
        return None


def _header(headers, name):
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _format_date(raw_date: str) -> str:
    if not raw_date:
        return "unknown date"
    try:
        return parsedate_to_datetime(raw_date).strftime("%a %b %d, %I:%M %p")
    except Exception:
        return raw_date


def _format_message_summary(service, msg_id: str) -> str:
    """Fetch one message's metadata (subject/from/date + labels) and format a
    single summary line. format='metadata' avoids pulling full bodies down;
    Gmail's own snippet field gives a short preview for free."""
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="metadata",
        metadataHeaders=["From", "Subject", "Date"],
    ).execute()

    headers = msg.get("payload", {}).get("headers", [])
    sender = _header(headers, "From") or "Unknown sender"
    subject = _header(headers, "Subject") or "(no subject)"
    date = _format_date(_header(headers, "Date"))
    snippet = msg.get("snippet", "").strip()
    unread = "UNREAD" in msg.get("labelIds", [])
    flag = "\U0001F535 " if unread else ""

    line = f"{flag}{date} \u2014 From: {sender}\n   Subject: {subject}"
    if snippet:
        line += f"\n   {snippet[:200]}"
    return line


def perform_gmail_check(payload: str = "") -> str:
    """Tool entry point.

    Empty payload -> most recent messages in the inbox.
    Non-empty payload -> treated as a raw Gmail search query (from:, subject:,
    is:unread, newer_than:7d, has:attachment, etc.) and passed through as-is.

    Strictly read-only: builds the API client with the gmail.readonly scope,
    so there is no code path here that can send, delete, or modify mail.
    """
    payload = (payload or "").strip()

    if not GMAIL_LIBS_AVAILABLE:
        return ("Error: Gmail libraries are not installed. Install them with: "
                "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

    cache_key = payload or "__inbox__"
    now = time.time()
    if _cache["key"] == cache_key and (now - _cache["time"]) < CACHE_TTL_SECONDS:
        log_event("gmail:cache_hit", {"query": cache_key})
        return _cache["result"]

    service = _get_service()
    if not service:
        return (
            "Error: Gmail is not authorized. Place your OAuth client secret at "
            f"'{CREDENTIALS_FILE}' (from Google Cloud Console, Desktop app type), make sure the "
            "Gmail API libraries are installed, and try again — the first attempt will open a "
            "browser window for one-time consent."
        )

    try:
        list_kwargs = {"userId": "me", "maxResults": DEFAULT_MAX_RESULTS}
        if payload:
            list_kwargs["q"] = payload
        else:
            list_kwargs["labelIds"] = CONFIG.get("gmail", "label_ids", default=["INBOX"])

        response = service.users().messages().list(**list_kwargs).execute()
        message_refs = response.get("messages", [])

        if not message_refs:
            result = "No matching emails found." if payload else "Inbox is empty (or nothing recent)."
            _cache.update({"key": cache_key, "time": now, "result": result})
            return result

        header_line = f"Gmail search results for: '{payload}'" if payload else f"Recent inbox messages (up to {DEFAULT_MAX_RESULTS}):"
        lines = [header_line]
        for ref in message_refs:
            try:
                lines.append("- " + _format_message_summary(service, ref["id"]))
            except Exception as e:
                log_event("gmail:message_fetch_error", {"id": ref.get("id"), "error": str(e)})
                continue

        result = "\n".join(lines)
        _cache.update({"key": cache_key, "time": now, "result": result})
        log_event("gmail:query", {"payload": payload, "count": len(message_refs)})
        return result

    except Exception as e:
        log_event("gmail:api_error", str(e))
        return f"Error querying Gmail: {e}"
