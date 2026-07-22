"""Central configuration.

Non-secret, tunable settings live in config.json (see the copy at the project
root for every available field). Secrets — TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID — stay in .env and are never read from or written to
config.json. Gmail's OAuth client secret and cached token are separate
files on disk (see gmail.credentials_file / gmail.token_file below) — not
env vars — since that's the format Google's own libraries expect, but
they're blocked from the read_file tool the same way .env is (see
security.file_access_blocked_names).

Usage elsewhere in the codebase:

    from jarvis.config import CONFIG
    model = CONFIG.get("ollama", "model")
    timeout = CONFIG.get("ollama", "connect_timeout", default=10)
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH_ENV_VAR = "JARVIS_CONFIG_PATH"

# These mirror config.json exactly. If config.json is missing (or missing a
# key), the assistant still runs with these values rather than crashing.
DEFAULT_CONFIG = {
    "assistant": {
        "name": "Jarvis",
        "host_name": "Noah",
        "system_prompt_version": "jarvis_v1.8",
    },
    "features": {
        "web_search": True,
        "shell_exec": True,
        "environment_tool": True,
        "peek": True,
        "cam_peek": True,
        "code_tools": True,
        "soul_memory": True,
        "dj_mode": True,
        "telegram": True,
        "gmail": True,
        "school_calendar": True,
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "model": "huihui_ai/gemma-4-abliterated",
        "connect_timeout": 10,
        "read_timeout": 45,
        "temperature": 0.7,
        "model_context": 131072,
    },
    "turns": {
        "max_autonomous_turns": 12,
        "max_tool_iterations": 12,
        "autopilot_max_turns": 8,
        "reset_confirm_window_seconds": 120,
    },
    "memory": {
        "context_threshold": 0.75,
        "warning_threshold": 0.50,
        "condense_threshold": 0.75,
        "recent_turns_to_keep": 8,
        "char_budget_multiplier": 4,
        "summary_char_limit": 1800,
        "check_interval_prompts": 3,
        "trim_max_turns": 20,
    },
    "files": {
        "conversation_history": "conversation_history.json",
        "soul": "soul.json",
        "interaction_log": "interaction_log.txt",
        "tts_settings": "tts_settings.json",
    },
    "telegram": {
        "poll_timeout": 30,
        "stream_tool_calls": True,
        "images_dir": "telegram_images",
        "audio_dir": "telegram_audio",
        "files_dir": "telegram_files",
        "voice_out_dir": "telegram_voice_replies",
    },
    "webcam": {
        "captures_dir": "telegram_webcam",
        "cam_index": 0,
        "yolo_weights": "yolov8n.pt",
    },
    "vosk": {
        "model_path": "models/vosk-model-small-en-us-0.15",
    },
    "tts": {
        "rate": 155,
        "volume": 0.7,
    },
    "web_search": {
        "max_results": 3,
        "cache_ttl_seconds": 30,
        "blocked_keywords": ["Natal", "Noah Smith", "your search query here", "<query>"],
    },
    "gmail": {
        # OAuth client secret downloaded from Google Cloud Console (Desktop
        # app type). Never commit this file.
        "credentials_file": "gmail_credentials.json",
        # Cached OAuth token, written automatically after the first
        # successful browser consent flow. Never commit this file either.
        "token_file": "gmail_token.json",
        "max_results": 10,
        "cache_ttl_seconds": 60,
        "label_ids": ["INBOX"],
    },
    "shell": {
        "default_timeout_seconds": 15,
        "blocked_patterns": ["rm -rf", "format", "dd if", "> /dev", "mkfs", "shred"],
        "command_translations": {
            "ls": "dir",
            "ls -la": "dir /a",
            "ls -l": "dir",
            "pwd": "echo %cd%",
            "whoami": "whoami",
        },
    },
    "environment_probe": {
        "commands": [
            ["whoami", "Current user"],
            ["echo %cd%", "Current working directory"],
            ["ver", "Windows version"],
            ["dir", "Files in current directory"],
            ["nvidia-smi", "GPU info"],
            ["systeminfo | findstr /B /C:\"OS Name\" /C:\"OS Version\"", "OS details"],
        ],
    },
    "desktop_peek": {
        "timeout_seconds": 30,
        "excluded_terms": [
            "N/A", "OleMainThreadWndName", "OLEChannelWnd", "Search", "NVIDIA",
            ".NET", "Start", "Windows", "nvcontainer.exe", "Task Host Window",
            "Settings", "Notification", "nvsphelper64", "RtkAudUService64",
            "lghub_system_tray.exe", "Discord Overlay Input Trap",
        ],
    },
    "text_file_extensions": [".txt", ".md", ".csv", ".json", ".log", ".py", ".yaml", ".yml", ".ini", ".cfg"],
    "security": {
        # Defense-in-depth for code_dev/code_exec: these are checked against the
        # *contents* of a Python file (not just shell_exec commands), since a
        # model could otherwise write a script that shells out or deletes files
        # and route around the shell_exec blocklist entirely.
        "code_blocked_patterns": [
            "os.system(", "os.popen(", "subprocess.", "shutil.rmtree(",
            "os.remove(", "os.unlink(", "os.rmdir(", "eval(", "exec(",
            "__import__(", "ctypes.", "socket.socket(",
        ],
        # read_file / ingest_image / transcribe_voice all accept an arbitrary
        # path straight from model output. Confine them to these roots
        # (resolved absolute paths must be inside one of these, or the request
        # is refused) so a crafted "../../" or absolute path can't walk the
        # model outside the assistant's own working area.
        "file_access_allowed_roots": [
            ".", "telegram_files", "telegram_images", "telegram_audio",
            "telegram_voice_replies", "telegram_webcam",
        ],
        # Blocked by filename regardless of which allowed root it's under.
        # gmail_token.json grants live read access to the inbox and
        # gmail_credentials.json is the OAuth client secret — both stay off
        # limits to read_file exactly like .env.
        "file_access_blocked_names": [".env", "gmail_token.json", "gmail_credentials.json"],
    },
    "debug": {
        "ollama": False,
        "path": False,
        "dj": False,
        "cam": False,
        "tts": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto base, returning a new dict. Lists and
    scalars in override replace the base value outright (not merged)."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    def __init__(self, path: str = "config.json"):
        self.path = os.environ.get(CONFIG_PATH_ENV_VAR, path)
        self._data = DEFAULT_CONFIG
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                self._data = _deep_merge(DEFAULT_CONFIG, user_config)
            except Exception as e:
                print(f"Warning: could not load {self.path} ({e}); using built-in defaults.")
        else:
            print(f"No {self.path} found — using built-in defaults.")

        # Secrets: .env only, never config.json.
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def get(self, *keys, default=None):
        """CONFIG.get('memory', 'recent_turns_to_keep', default=8)"""
        node = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def feature_enabled(self, name: str) -> bool:
        return bool(self.get("features", name, default=True))

    def as_dict(self) -> dict:
        return self._data


CONFIG = Config()
