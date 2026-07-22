"""One-time detection of optional external dependencies (ffmpeg, yt-dlp,
opencv, ultralytics, Gmail API libs), shared by every module that needs to
know whether a given tool can actually run. Computed once at import time,
same as the original script."""
import importlib.util
import shutil
import subprocess
import sys

FFMPEG_PATH = shutil.which("ffmpeg")
FFMPEG_AVAILABLE = FFMPEG_PATH is not None  # required for voice-note <-> wav conversion

YTDLP_PATH = shutil.which("yt-dlp")
YTDLP_AVAILABLE = YTDLP_PATH is not None  # required for DJ mode (dj_play / dj_stop)

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None  # required for cam_peek
YOLO_AVAILABLE = importlib.util.find_spec("ultralytics") is not None  # required for cam_peek

GMAIL_LIBS_AVAILABLE = (
    importlib.util.find_spec("googleapiclient") is not None
    and importlib.util.find_spec("google_auth_oauthlib") is not None
)  # required for the gmail tool (read-only inbox listing/search)


def diagnose_ffmpeg():
    """Print exactly what this Python process sees, to distinguish 'PATH is
    stale in this process' from 'ffmpeg is found but fails to run' from
    'genuinely not on PATH here'."""
    print("---- ffmpeg diagnostics ----")
    print(f"Python executable: {sys.executable}")
    which_result = shutil.which("ffmpeg")
    print(f"shutil.which('ffmpeg'): {which_result or 'NOT FOUND'}")

    import os
    path_env = os.environ.get("PATH", "")
    path_entries = [p for p in path_env.split(os.pathsep) if p.strip()]
    ffmpeg_like = [p for p in path_entries if "ffmpeg" in p.lower()]
    print(f"PATH entries containing 'ffmpeg' ({len(ffmpeg_like)}):")
    for p in ffmpeg_like:
        print(f"   {p}")
    if not ffmpeg_like:
        print("   (none — this process's PATH does not include an ffmpeg folder at all)")

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=8)
        first_line = (result.stdout or result.stderr or "").splitlines()[0] if (result.stdout or result.stderr) else ""
        print(f"Direct 'ffmpeg -version' call: return code {result.returncode} — {first_line}")
    except FileNotFoundError:
        print("Direct 'ffmpeg -version' call: FileNotFoundError — this process genuinely cannot resolve ffmpeg on PATH.")
    except Exception as e:
        print(f"Direct 'ffmpeg -version' call: raised {type(e).__name__}: {e}")
    print("-----------------------------")
    if not which_result:
        print("If cmd.exe can run ffmpeg but this script can't, the process running this")
        print("script (terminal, IDE, or launcher) was almost certainly started BEFORE")
        print("you added ffmpeg to PATH, and is holding onto the old PATH value. Fully")
        print("close that terminal/IDE (not just the tab) and reopen it, then rerun.")
        print("Also check whether you edited the USER PATH vs SYSTEM PATH, and whether")
        print("this process is running elevated (Run as Administrator) vs not — those")
        print("environments can differ.")
