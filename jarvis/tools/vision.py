"""Desktop-awareness tools: `peek` (filtered snapshot of what's running on
the host desktop) and `cam_peek` (single webcam frame + YOLOv8n person
detection, pushed to Telegram when the triggering turn came in remotely)."""
import base64
import os
import time

from jarvis.capabilities import CV2_AVAILABLE, YOLO_AVAILABLE
from jarvis.config import CONFIG
from jarvis.logging_utils import log_event
from jarvis.telegram_bot import telegram_send_photo
from jarvis.tools.shell import shell_exec

WEBCAM_CAPTURES_DIR = CONFIG.get("webcam", "captures_dir", default="telegram_webcam")
os.makedirs(WEBCAM_CAPTURES_DIR, exist_ok=True)


# ====================== DESKTOP AWARENESS: PEEK TOOL ======================
# Filtered snapshot of what's actually running/visible on the host desktop right now.
# The findstr chain strips out OS plumbing and background helper processes that are
# never useful signal — tune the excluded term list in config.json -> desktop_peek.
def build_peek_command() -> str:
    excluded_terms = CONFIG.get("desktop_peek", "excluded_terms", default=[])
    command = 'tasklist /v /fi "status eq running"'
    for term in excluded_terms:
        command += f' | findstr /v "{term}"'
    return command


def perform_peek() -> str:
    """Run the filtered tasklist snapshot in the background and return a clean
    summary for the LLM — lets Jarvis 'glance' at what's running on the host's
    desktop without the user having to describe it."""
    print("👀 Peeking at the host desktop...")
    log_event("peek:start")

    timeout = CONFIG.get("desktop_peek", "timeout_seconds", default=30)
    stdout, stderr, rc = shell_exec(build_peek_command(), timeout=timeout)

    if stdout:
        output = "=== CURRENTLY RUNNING ON HOST DESKTOP (filtered) ===\n\n" + stdout.strip()
    elif stderr:
        output = f"Peek failed: {stderr.strip()}"
    else:
        output = "Peek returned no output (nothing left after filtering, or tasklist produced nothing)."

    log_event("peek:results", output)
    return output


# ====================== DESKTOP AWARENESS: CAM_PEEK TOOL ======================
# Single webcam frame -> YOLOv8n person detection (presence/count only, no
# identification of who it is) -> image + detection summary handed to the model,
# and pushed to Telegram when the triggering turn came in over Telegram.
_yolo_model = None  # module-level cache so the model is loaded once per process, not once per call


def _get_yolo_model():
    """Lazily import ultralytics and load the configured YOLO weights. Kept
    lazy (rather than a top-level import) so a script restart/startup doesn't
    pay the torch import and model-load cost unless cam_peek is actually
    used. First call may also trigger a one-time download of the weights."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(CONFIG.get("webcam", "yolo_weights", default="yolov8n.pt"))
    return _yolo_model


def _capture_webcam_frame(cam_index: int = None):
    """Grab a single frame from the desktop webcam. Returns (frame, error)."""
    import cv2
    cam_index = cam_index if cam_index is not None else CONFIG.get("webcam", "cam_index", default=0)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # DSHOW backend avoids slow MSMF init on Windows
    if not cap.isOpened():
        cap.release()
        return None, "Could not open the webcam — it may be in use by another app, disabled, or not present."
    frame, ok = None, False
    for _ in range(5):
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, "Webcam opened but failed to capture a frame."
    return frame, None


def perform_cam_peek(telegram_chat_id: str = None) -> dict:
    """Capture a webcam frame, run YOLOv8n person detection on it, save it to
    the configured webcam captures dir, and push it to Telegram if this
    request came in over Telegram. Returns {"ok", "summary", "base64", "path"}
    for the tool loop."""
    print("📷 Capturing webcam frame for cam_peek...")
    log_event("cam_peek:start")

    if not CV2_AVAILABLE:
        msg = "cam_peek unavailable: opencv-python is not installed (`pip install opencv-python`)."
        log_event("cam_peek:unavailable", "cv2")
        return {"ok": False, "summary": msg, "base64": None, "path": None}
    if not YOLO_AVAILABLE:
        msg = "cam_peek unavailable: ultralytics is not installed (`pip install ultralytics`)."
        log_event("cam_peek:unavailable", "ultralytics")
        return {"ok": False, "summary": msg, "base64": None, "path": None}

    frame, err = _capture_webcam_frame()
    if err:
        log_event("cam_peek:capture_error", err)
        return {"ok": False, "summary": f"Webcam capture failed: {err}", "base64": None, "path": None}

    import cv2
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(WEBCAM_CAPTURES_DIR, f"{timestamp}_cam_peek.jpg")
    cv2.imwrite(filepath, frame)

    try:
        model = _get_yolo_model()
        results = model(frame, verbose=False)
        confidences = [
            float(box.conf[0])
            for r in results
            for box in r.boxes
            if model.names.get(int(box.cls[0])) == "person"
        ]
    except Exception as e:
        log_event("cam_peek:yolo_error", str(e))
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        summary = f"Captured a webcam frame, but person-detection failed: {e}"
        if telegram_chat_id:
            telegram_send_photo(telegram_chat_id, filepath, caption="📷 Webcam capture (detection failed)")
        log_event("cam_peek:results", {"summary": summary, "path": filepath})
        return {"ok": True, "summary": summary, "base64": b64, "path": filepath}

    count = len(confidences)
    if count == 0:
        summary = "No person detected in the webcam frame."
    elif count == 1:
        summary = f"1 person detected in the webcam frame (confidence {confidences[0]:.2f})."
    else:
        conf_str = ", ".join(f"{c:.2f}" for c in sorted(confidences, reverse=True))
        summary = f"{count} people detected in the webcam frame (confidences: {conf_str})."

    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    if telegram_chat_id:
        sent = telegram_send_photo(telegram_chat_id, filepath, caption=f"📷 {summary}")
        if not sent:
            summary += " (Note: failed to deliver the photo to Telegram.)"

    log_event("cam_peek:results", {"summary": summary, "path": filepath, "telegram_sent": bool(telegram_chat_id)})
    return {"ok": True, "summary": summary, "base64": b64, "path": filepath}
