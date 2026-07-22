"""Shell execution (the most powerful/dangerous tool), Python code
write/run tools, and environment discovery — all built on the same raw
shell_exec primitive. Windows/cmd.exe oriented, same as the original script;
the command list and translations are configurable via config.json."""
import os
import subprocess
import time

from jarvis.config import CONFIG
from jarvis.logging_utils import log_event


def kill_process_tree(proc, timeout: int = 3) -> bool:
    """Terminate `proc` and any child processes it spawned.

    subprocess's own terminate()/kill() only signal the direct child. On
    Windows that's not enough: a `shell_exec` command that launches another
    process (or a Popen pair like DJ mode's yt-dlp -> ffplay) can leave
    grandchildren running as orphans after we think we've stopped it.
    `taskkill /T` walks the whole process tree; if taskkill itself isn't
    available (e.g. developing on a non-Windows box) this falls back to a
    plain terminate()/kill() on just the one process, same as before.
    """
    if proc is None or proc.poll() is not None:
        return False
    try:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True, text=True, timeout=timeout
        )
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    # taskkill unavailable or didn't finish in time — fall back to a direct kill.
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
    except Exception:
        pass
    return proc.poll() is not None


def shell_exec(command: str, timeout: int = None) -> tuple:
    """Raw shell executor (this is by far the most powerful (or dangerous) tool).
    Uses Popen directly (rather than subprocess.run) so that on a timeout we
    have a handle to kill the *whole* process tree via kill_process_tree,
    instead of only the immediate cmd.exe child."""
    timeout = timeout if timeout is not None else CONFIG.get("shell", "default_timeout_seconds", default=15)
    proc = None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout.strip(), stderr.strip(), proc.returncode
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=2)
        except Exception:
            pass
        return "", f"Command timed out after {timeout} seconds (process tree terminated)", -1
    except Exception as e:
        if proc is not None:
            kill_process_tree(proc)
        return "", f"Execution error: {str(e)}", 1


def _is_shell_command_blocked(command: str) -> str:
    """Return a block reason if `command` matches a configured destructive
    pattern, else "". Centralized here — inside shell.py itself — rather than
    only in the tool-dispatch loop, so every caller of perform_shell_exec
    gets the same protection instead of relying on each call site to
    remember to check first."""
    blocked_patterns = CONFIG.get("shell", "blocked_patterns", default=[])
    lowered = command.lower()
    for pattern in blocked_patterns:
        if pattern.lower() in lowered:
            return f"matched blocked pattern '{pattern}'"
    return ""


def perform_shell_exec(command: str) -> str:
    """Wrapper for shell_exec that formats the output nicely for TTS response.
    Also the single enforcement point for the empty-command guard and the
    destructive-pattern blocklist, so this function is safe to call from
    anywhere (not just the one tag-dispatch call site that used to do the
    check itself)."""
    command = (command or "").strip()
    if not command:
        log_event("shell_exec:blocked", {"reason": "empty_command"})
        return "Shell command executed: ``\nReturn code: -1\n\nSTDERR:\nNo command provided in shell request."

    blocked_reason = _is_shell_command_blocked(command)
    if blocked_reason:
        log_event("shell_exec:blocked", {"reason": "destructive", "command": command, "detail": blocked_reason})
        return (
            f"Shell command executed: `{command}`\n"
            "Return code: -1\n\n"
            f"STDERR:\nBlocked: command {blocked_reason} and was not executed."
        )

    translations = CONFIG.get("shell", "command_translations", default={})
    cmd_lower = command.lower()
    if cmd_lower in translations:
        command = translations[cmd_lower]
        print(f"🔄 Translated to Windows command: {command}")
    print(f"🛠️  Executing: {command}")

    log_event("shell_exec:request", command)

    stdout, stderr, returncode = shell_exec(command)

    output = f"Shell command executed: `{command}`\n"
    output += f"Return code: {returncode}\n\n"

    if stdout:
        output += f"STDOUT:\n{stdout}\n\n"
    if stderr:
        output += f"STDERR:\n{stderr}\n\n"

    output = output.strip()
    log_event("shell_exec:results", output)
    return output


def _scan_code_for_blocked_patterns(code: str) -> str:
    """Return a block reason if `code` contains a pattern from config.json ->
    security.code_blocked_patterns, else "". Defense in depth against
    code_dev/code_exec being used to route around the shell_exec blocklist —
    e.g. writing a Python script that shells out or wipes files instead of
    calling shell_exec directly. Applied both when a file is written
    (code_dev) and again right before it's run (code_exec), since the file
    on disk could have been created or edited some other way in between."""
    blocked_patterns = CONFIG.get("security", "code_blocked_patterns", default=[])
    lowered = code.lower()
    for pattern in blocked_patterns:
        if pattern.lower() in lowered:
            return f"matched blocked pattern '{pattern}'"
    return ""


def _is_safe_relative_filename(filename: str) -> bool:
    """Reject absolute paths and '..' traversal so code_dev/code_exec stay
    confined to the working directory tree."""
    if not filename:
        return False
    if os.path.isabs(filename):
        return False
    normalized = os.path.normpath(filename)
    if normalized.startswith("..") or normalized.startswith(os.sep):
        return False
    return True


def perform_code_dev(payload: str) -> str:
    """Write or develop code files without executing them.
    Payload format:
    language
    optional_filename.py
    full code here (multi-line is fine)
    """
    try:
        lines = payload.strip().splitlines()
        if len(lines) < 2:
            return "Error: code_dev needs at least language and code."

        language = lines[0].strip().lower()
        if language != "python":
            return f"Only python supported right now, got: {language}"

        if lines[1].strip().endswith(".py"):
            filename = lines[1].strip()
            code_start = 2
        else:
            filename = f"temp_jarvis_{int(time.time())}.py"
            code_start = 1

        if not _is_safe_relative_filename(filename):
            log_event("code_dev:blocked", {"filename": filename, "reason": "unsafe_path"})
            return f"Error: '{filename}' is not a safe filename (no absolute paths or '..' allowed)."

        code = "\n".join(lines[code_start:]).strip()

        blocked_reason = _scan_code_for_blocked_patterns(code)
        if blocked_reason:
            log_event("code_dev:blocked", {"filename": filename, "detail": blocked_reason})
            return (
                f"Blocked: refused to write {filename} — code {blocked_reason}. "
                "If this is genuinely needed, ask the user to run it manually instead."
            )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        output = (
            f"✅ Code file created ({filename})\n"
            f"Language: {language}\n"
            f"Characters written: {len(code)}\n"
            "Execution skipped by design (use <code_exec> to run)."
        )
        log_event("code_dev:results", output)
        return output

    except Exception as e:
        log_event("code_dev:error", str(e))
        return f"Code development error: {e}"


def perform_code_exec(payload: str) -> str:
    """Execute existing code only.
    Payload format:
    python
    existing_filename.py
    optional stdin lines...
    """
    try:
        lines = payload.strip().splitlines()
        if len(lines) < 2:
            return "Error: code_exec requires language and an existing filename to run."

        language = lines[0].strip().lower()
        if language != "python":
            return f"Only python supported right now, got: {language}"

        filename = lines[1].strip()
        if not filename.endswith(".py"):
            return "Error: code_exec requires an existing .py filename on line 2."
        if not _is_safe_relative_filename(filename):
            log_event("code_exec:blocked", {"filename": filename, "reason": "unsafe_path"})
            return f"Error: '{filename}' is not a safe filename (no absolute paths or '..' allowed)."
        if not os.path.exists(filename):
            return f"Error: file not found for execution: {filename}"

        try:
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                existing_code = f.read()
        except Exception as e:
            return f"Error: could not read {filename} before execution: {e}"

        blocked_reason = _scan_code_for_blocked_patterns(existing_code)
        if blocked_reason:
            log_event("code_exec:blocked", {"filename": filename, "detail": blocked_reason})
            return f"Blocked: refused to run {filename} — its contents {blocked_reason}."

        stdin_payload = "\n".join(lines[2:]).strip()
        stdin_input = f"{stdin_payload}\n" if stdin_payload else None

        print(f"▶️ Running existing script: {filename}")

        result = subprocess.run(
            ["python", filename],
            capture_output=True,
            text=True,
            timeout=30,
            input=stdin_input,
            cwd=os.getcwd()
        )

        output = f"✅ Code execution complete ({filename})\n"
        output += f"Return code: {result.returncode}\n\n"
        if result.stdout.strip():
            output += f"STDOUT:\n{result.stdout.strip()}\n\n"
        if result.stderr.strip():
            output += f"STDERR:\n{result.stderr.strip()}\n\n"

        log_event("code_exec:results", output)
        return output

    except Exception as e:
        log_event("code_exec:error", str(e))
        return f"Code execution error: {e}"


def perform_get_environment() -> str:
    """Run safe diagnostic commands and return a clean summary for the LLM."""
    print("🔍 Gathering environment context for Jarvis...")
    log_event("get_environment:start")

    commands = CONFIG.get("environment_probe", "commands", default=[])

    output = "=== ENVIRONMENT CONTEXT ===\n\n"

    for cmd, label in commands:
        stdout, stderr, rc = shell_exec(cmd, timeout=8)
        if stdout:
            output += f"{label}:\n{stdout.strip()}\n\n"
        elif stderr:
            output += f"{label}: (error) {stderr.strip()}\n\n"

    output += "You are running on Windows (cmd.exe). Use 'dir' instead of 'ls', 'echo %cd%' instead of 'pwd', etc.\n"
    output += "You can now use <shell_exec> with Windows-native commands."

    log_event("get_environment:results", output)
    return output
