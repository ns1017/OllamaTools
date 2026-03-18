# OllamaTools

OllamaTools is a personal-assistant project that expands local LLMs that do not include native tool use by default.

This repository provides an expansion to models that don't come with tool capabilities by default with a personal assistant focus.
It also works with tools that are built with tool calling, possibly making tool triggers cleaner.

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
- Limited autonomous continuation with loop and safety guards

## Known Issues

- Probably the most signifigant, the AI rarely is able to correct itself while in an autonomous session. It ends up looping the incorrect function or a tool call incorrectly.
- While not incompatible with linux, there are built in filters to correct linux cli commands to windows, there may be hallucinations when "get_env" command are used without altering the system prompt or command filter.
- Some issues with text-to-speech. For example, sometimes switching back from keyboard input does not take precedence over the AI finishing its response. 

## Project Structure

- `voice_ollama_v1_release.py`: Main assistant runtime
- `models/`: Vosk speech model assets
- `conversation_history.json`: Persistent chat history, generated after first run.
- `interaction_log.txt`: Runtime and event logs, generated after first run.
- `app_settings.json`: Local app configuration

## Requirements

- Windows (current implementation targets cmd/Windows behavior)
- Python 3.10+
- Ollama installed and available in PATH
- A local Ollama model pulled and configured in the script, Ollama running
- A Vosk model folder in the directory at:
  - `models/vosk-model-small-en-us-0.15`
- A Vosk model configured if the default is not used

Python packages used by the main script:

- `sounddevice`
- `vosk`
- `requests`
- `pyttsx3`
- `ddgs`
- `keyboard`
- `python-dotenv`

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install sounddevice vosk requests pyttsx3 ddgs keyboard python-dotenv
```

3. Ensure Ollama is installed and running:

```powershell
ollama serve
```

4. Pull a model (huihui_ai/orchestrator-abliterated:8b used in personal testing):

```powershell
ollama pull huihui_ai/orchestrator-abliterated:8b
```

5. Configure your model in voice_ollama_v1_release.py

6. Place Vosk model file(s) at `models/vosk-model-small-en-us-0.15`, correct model version if changed.

## Running

From the project directory:

```powershell
python voice_ollama_v1_release.py
```

## Controls

During runtime:

- `m`: Toggle bot mute
- `i`: Toggle mic mute
- `q`: Quit
- `Ctrl+Shift+K`: Toggle speech/keyboard input mode

## Tooling Model

The assistant instructs the LLM to emit strict XML-like tags for tool routing:

- `<web_search>...</web_search>`
- `<shell_exec>...</shell_exec>`
- `<get_environment></get_environment>`
- `<code_dev>...</code_dev>`
- `<code_exec>...</code_exec>`
- `<continue>...</continue>`

These are parsed and executed in a guarded loop with iteration limits.

## Safety Notes

- Shell execution includes basic blocked-command checks but is still powerful.
- Python OS calls are ran at the same privalege level the initial script is ran at.
- This means running voice_ollama_v1_release.py as admin could be very bad.
- Review and harden allow/deny rules before ANY usage.

## Configuration Notes

- TTS defaults can be overridden with `tts_settings.json`.
- `.env` can be added for future Telegram integration but Telegram features are currently commented out.
- Conversation and event logs are written locally by default.

## Roadmap Ideas

- Wake word for speech-to-text
- DJ mode for playing and queing music 
- Stronger command sandboxing and allowlist-based shell policy
- Better context pruning and memory management
- More robust retry and tool-failure classification
- Cross-platform command abstraction beyond Windows
- Optional remote control integration (Telegram)

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the LICENSE file for the full license text.
