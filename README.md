# OllamaTools v2/Jarvis

The initial `voice_ollama_v1.py` (~3,100-lines) split into a
package, with every tunable value pulled into `config.json`.

## config.json reference

Every field has a built-in default (see `jarvis/config.py`), so
`config.json` technically optional. Any field you omit from
your `config.json` falls back to the default.

| Section | Purpose |
|---|---|
| `assistant` | Name, host name, and the system-prompt version tag |
| `features` | Turn whole tools on/off (`web_search`, `shell_exec`, `environment_tool`, `peek`, `cam_peek`, `code_tools`, `soul_memory`, `dj_mode`, `telegram`). Disabling a tool removes its instructions from the system prompt too, not just its dispatch. |
| `ollama` | `base_url`, `model`, timeouts, `temperature`, `model_context` |
| `turns` | Autonomy budgets: `max_autonomous_turns`, `max_tool_iterations`, `autopilot_max_turns`, `reset_confirm_window_seconds` |
| `memory` | Short-term memory pressure thresholds and condensation tuning |
| `files` | Paths for `conversation_history`, `soul`, `interaction_log`, `tts_settings` |
| `telegram` | Poll timeout, whether to stream tool-call status lines, media download directories |
| `webcam` | Capture directory, camera index, YOLO weights file |
| `vosk` | Path to the Vosk speech model |
| `tts` | Default rate/volume (per-run overrides still come from `tts_settings.json`) |
| `web_search` | Result count, cache TTL, blocked keywords |
| `shell` | Default timeout, destructive-command blocklist, `ls`/`pwd`-style translations |
| `environment_probe` | The `(command, label)` pairs `get_environment` runs |
| `desktop_peek` | Timeout and the list of process-name substrings filtered out of `peek` |
| `text_file_extensions` | Which extensions `read_file` will read as plain text |
| `debug` | Startup diagnostic print toggles (`ollama`, `path`, `dj`, `cam`, `tts`) |

<img width="864" height="1276" alt="1" src="https://github.com/user-attachments/assets/5efb18dd-42c1-48f0-b4ec-970410904abd" />

- Model name must be set
- Base URL may need changed

<img width="762" height="1275" alt="2" src="https://github.com/user-attachments/assets/80eb3dc0-db55-4197-bcd7-7c26bc6c10b8" />

-.ics link will need uploaded
- Excluded Terms will vary system to system

<img width="952" height="235" alt="3" src="https://github.com/user-attachments/assets/700c8829-5f64-4417-8cfb-1458c2b6d44d" />

- Individual debugs can be toggled if issues occur

To point at a different config file (e.g. per-machine configs), set
`JARVIS_CONFIG_PATH` before running:

```
JARVIS_CONFIG_PATH=config.laptop.json python main.py
```

## What It Does

OllamaTools runs a voice-first assistant loop with optional keyboard input and adds structured tool orchestration around an Ollama chat model.

Core capabilities include:

- Speech-to-text input with Vosk and microphone capture
- Text-to-speech output with pyttsx3
- Tool tag parsing from model responses
- Web search support (DuckDuckGo via ddgs)
- Shell command execution wrapper with basic safety checks
- Python code creation and execution helpers (`code_dev`, `code_exec`)
- Environment discovery tool for reliable OS-aware behavior
- Conversation persistence and interaction logging
- Limited & potentially unlimted autonomous continuation with loop and safety guards ( true autonomous/autopilot mode is now a seperate input)
- Full telegram implimentation for AFK use, accepts images and voice notes. Also responds with voice note if prompted with one
- Webcam discovery
- DJ mode/song queing
- long term memory
- real-time context reset
- possible gmail integration (advanced Oauth setup required)
- possible calander feed integration if a .ics feed

## Known Issues

- While not incompatible with linux, there are built in filters to correct linux cli commands to windows, there may be hallucinations when "get_env" commands are used without altering the system prompt or command filter.
- Issue with yt-dlp requests being limited/return bot verification requests occasionally. Cookies.txt doesn't always fix.

## Requirements

- Windows (current implementation targets cmd/Windows behavior)
- Python 3.10+
- Ollama installed and available in PATH; Download here: https://ollama.com/
- A local Ollama model pulled and configured in the script, Ollama running
- A Vosk model folder in the directory project directory like:
  - `models/vosk-model-small-en-us-0.15`
- Download a Vosk model here, I recommend the one above; models can be found here: https://alphacephei.com/vosk/models
- yolov8n.pt in the main project directory, download from here: https://docs.ultralytics.com/models/yolov8#performance-metrics
- ffmpeg installed to path (for djmode use): https://www.ffmpeg.org/download.html#build-windows
- External telegram bot configuration, place keys in .env

Python packages used by the main script:

- `sounddevice`
- `vosk`
- `requests`
- `pyttsx3`
- `ddgs`
- `keyboard`
- `python-dotenv`
- `open-cvpython`
- `ultralytics`
- `yt-dlp`
- `icalendar`
- `google-api-python-client`

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure Ollama is installed and running (or don't 😉  now starts automatically):

```powershell
ollama serve
```

4. Pull a model (huihui_ai/gemma4-abliterated:latest used in testing):

```powershell
ollama pull huihui_ai/gemma4-abliterated:latest
```

## Running

From the project directory:

```powershell
python main.py
```

## Controls

During runtime:

- `m`: Toggle bot mute
- `i`: Toggle mic mute
- `q`: Quit
- `Ctrl+Shift+K`: Toggle speech/keyboard input mode *Note that keyboard input will disable the commands above until toggled again.
- `Ctrl+Shift+T`: Toggle telegram streaming (disables other inputs for saftey)
- `Ctrl+Shift+A`: Toggle Autonomous mode. Toggle the mode first, then prompt.

## Tooling Model

The assistant instructs the LLM to emit strict XML-like tags for tool routing:

- `<web_search>...</web_search>`
- `<shell_exec>...</shell_exec>`
- `<get_environment></get_environment>`
- `<code_dev>...</code_dev>`
- `<code_exec>...</code_exec>`
- `<continue>...</continue>`
- `<soul_write>...</soul_write>` - acts as a unique long term memory that it semi hidden from the user. it's loaded at the start of every new context like the system prompt. Technically, the AI could prompt inject a robot revolution without the user ever knowing.

These are parsed and executed in a guarded loop with iteration limits.

## Safety Notes

- Shell execution includes basic blocked-command checks but is still powerful.
- Python OS calls are ran at the same privalege level the initial script is ran at.
- Review and harden allow/deny rules before ANY usage.

## Roadmap Ideas

- More robust retry and tool-failure classification
- Basic front-end
- Cross-platform command abstraction beyond Windows
- impliment an audio pipeline for models that support audio injestion for desktop/video and sound files (as opposed to relying on whisper transcription for everything).

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the LICENSE file for the full license text.
