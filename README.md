# OllamaTools

OllamaTools is a personal-assistant project that expands local LLMs that do not include native tool use by default.
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
- Limited & potentially unlimted autonomous continuation with loop and safety guards (know true autonomous/autopilot mode is now a seperate input)

## Known Issues

- While not incompatible with linux, there are built in filters to correct linux cli commands to windows, there may be hallucinations when "get_env" commands are used without altering the system prompt or command filter.
- Some issues with text-to-speech. For example, sometimes switching back from keyboard input does not take precedence over the AI finishing its response. 
- Issue occurs when a new chat is started. For some reason (most likely prompted by the ai), a continuation hides the inital AI response from the chat stream and only shows the final response. It can be viewed in the context history though.

## Project Structure

- `voice_ollama_v1_release.py`: Main assistant runtime
- `models/`: Vosk speech model assets
- `conversation_history.json`: Persistent chat history, generated after first run.
- `interaction_log.txt`: Runtime and event logs, generated after first run.
- `app_settings.json`: Local app configuration
- `.env`: contains sensitive keys/tokens that should generally be hidden from the AI. Now used for telegram configuration.

## Requirements

- Windows (current implementation targets cmd/Windows behavior)
- Python 3.10+
- Ollama installed and available in PATH; Download here: https://ollama.com/
- A local Ollama model pulled and configured in the script, Ollama running
- A Vosk model folder in the directory project directory like:
  - `models/vosk-model-small-en-us-0.15`
- Download a Vosk model here, I recommend the one above; models can be found here: https://alphacephei.com/vosk/models

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

4. Pull a model (huihui_ai/gemma4-abliterated:latest used in personal testing):

```powershell
ollama pull huihui_ai/gemma4-abliterated:latest
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
- `<soul_write>...</soul_write> - acts as a unique long term memory that it semi hidden from the user. it's loaded at the start of every new context like the system prompt. Technically, the AI could prompt inject a robot revolution without the user ever knowing.

These are parsed and executed in a guarded loop with iteration limits.

## Safety Notes

- Shell execution includes basic blocked-command checks but is still powerful.
- Python OS calls are ran at the same privalege level the initial script is ran at.
- This means running voice_ollama_v1_release.py as admin could be very bad.
- Review and harden allow/deny rules before ANY usage.

## Configuration Notes

- TTS defaults can be overridden with `tts_settings.json`.
- `.env` can be added for Telegram integration but Telegram features aren't fully implimented. i.e sending images, voice recordings, and files
- Conversation and event logs are written locally by default.

## Roadmap Ideas

- DJ mode for playing and queing music 
- Stronger command sandboxing and allowlist-based shell policy
- More robust retry and tool-failure classification
- Cross-platform command abstraction beyond Windows
- Finish remote control integration via Telegram (send/recieve voice is the goal).
- Image reading (for vision-language models)
- impliment an audio pipeline for models that support audio injestion for desktop/video and sound files (as opposed to relying on whisper transcription for everything).

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the LICENSE file for the full license text.
