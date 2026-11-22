# SessionMind

AI meeting intelligence. It listens to a meeting (browser screen/tab capture with audio, or an uploaded
recording), transcribes it, and produces a report with a summary, **decisions**, **action items**,
**open questions**, risks and a mind map, as an in-app view, a Word file and an HTML file.

Speech-to-text and the AI analysis can run in the **cloud (Groq)** or **fully on your machine
(Whisper + Ollama)**. You choose in `.env`; nothing else changes.

## Quick start

```bash
./run.sh            # creates the venv, installs, creates .env from .env.example, starts the app
```

Open http://127.0.0.1:8000. The status chip at the top tells you what is missing.

You need Python 3.10+ and FFmpeg (`sudo dnf install ffmpeg` · `sudo apt install ffmpeg` · `brew install ffmpeg`).

### Option A: cloud (easiest)

In `.env` set `GROQ_API_KEY=...`. That's all; Groq is the default for all three engines.

### Option B: fully local (nothing leaves your machine)

1. Install [Ollama](https://ollama.com) and pull a model: `ollama pull qwen2.5:7b-instruct`
2. Install local Whisper: `./run.sh --local` (or `pip install -r requirements-local.txt`)
3. In `.env`:
   ```
   LLM_PROVIDER=ollama
   STT_PROVIDER=whisper          # or faster_whisper
   WHISPER_MODEL=small
   WHISPER_LANGUAGE=en           # set your language; more accurate than auto-detect
   ```
4. Restart. The chip shows **Ready · fully local**.

You can also mix: for example local Whisper with Groq analysis.

## Choosing your AI engines

| Setting | Values | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq`, `ollama` | Analysis: finds items, writes the report |
| `STT_PROVIDER` | `groq`, `whisper`, `faster_whisper` | `whisper` = the official `openai-whisper` package. `faster_whisper` = same models, usually several times faster and a lighter install |
| `VISION_PROVIDER` | `auto`, `groq`, `ollama`, `none` | Reads captions/slides from screenshots. `auto` follows `LLM_PROVIDER`, so a local setup never sends images to the cloud. With Ollama it stays off until you set `OLLAMA_VISION_MODEL` |

Every setting is documented in `.env.example`.

### Practical advice for local models

These are rules of thumb, not benchmarks. Try them on your machine.

* **Whisper must keep up with the meeting.** If a 20-second chunk takes longer than 20 seconds to transcribe,
  the backlog grows. Use a smaller `WHISPER_MODEL` (`base`/`small` on CPU), `faster_whisper`, or a GPU.
  Uploading a recording after the meeting avoids the real-time constraint.
* **Bigger LLM, better extraction.** A 7-8B model is a reasonable start; 14B and above is noticeably better at
  assigning owners and not inventing items. Measure it (see *Evaluating a model*) rather than guessing.
* **Context size matters.** `OLLAMA_NUM_CTX` defaults to 16384 here because Ollama's own default is small and
  silently truncates long prompts. Raise it if you have memory; the report then carries more of a long meeting.
* **Whisper and the LLM share your GPU/RAM.** Running both at once is the usual cause of slowness.

## How decisions, action items and open questions are found

The earlier version missed items for specific, fixable reasons. The current pipeline (`app/services/extraction.py`):

1. **Small overlapping windows.** The transcript is read in slices of a few thousand characters, each with
   a few lines of context before it (so "Sure, I'll do it" is tied to the earlier request) and after it
   (so a question answered right away is not reported as open).
2. **Cue hints.** Lines that look like a question or a task are flagged for the model to check.
3. **Grounding.** Every item must point at a real transcript line. Its timestamp and evidence quote are taken
   from that line, and items that aren't in the transcript are discarded.
4. **No silent loss.** If the model fails on a slice, a keyword fallback still records the obvious questions and
   requests, marked *Unverified*, and the report shows a warning saying how many sections were affected.
5. **Merging in code.** Items from all slices and from the live panel are merged and de-duplicated (rephrased
   duplicates are recognised; different owners stay separate). The report model writes only the narrative, so a
   long meeting can no longer produce a cut-off JSON reply that fails the whole report. If even the narrative
   step fails, you still get a report with every extracted item.
6. **Long meetings** send the model per-section summaries instead of chopping off the beginning.

Items with no named owner get the status *Needs owner*; open questions are *Unanswered*, *Partially answered*,
*Deferred* or *Unverified*.

## Evaluating a model

```bash
python scripts/eval_extraction.py              # built-in labelled sample meeting
python scripts/eval_extraction.py my.json -v   # your own labelled transcript
```

It uses the engine configured in `.env` and prints recall per category and what was missed. Use it to compare
models before you change your default. Format of the labelled file: see `tests/data/sample_meeting.json`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use scripted stand-ins for the AI engines (and a local fake Ollama server), so they check the pipeline,
not model quality. They run without any API key, model or GPU.

## Project layout

```
app/
  main.py                 app wiring and /api/status
  config.py  db.py        settings; SQLite (schema created once, migrations applied once)
  providers/              the only place that talks to AI engines
    base.py               interfaces, JSON parsing, Whisper hallucination filter
    groq.py               cloud LLM + speech-to-text + vision
    ollama.py             local LLM + vision
    stt_local.py          openai-whisper and faster-whisper
  services/
    extraction.py         windows, prompts, grounding, fallback (no DB code)
    live.py               live insights while the meeting runs
    report.py             final report
    transcripts.py screenshots.py materials.py audio.py
  routes/                 HTTP endpoints (URLs unchanged)
  report_render.py        Word / HTML rendering, de-duplication
static/                   the web UI
scripts/eval_extraction.py
tests/
```

To add another engine (for example LM Studio or whisper.cpp), implement `LLMProvider` / `STTProvider` from
`providers/base.py` and register it in `providers/__init__.py`.

## Troubleshooting

* **"Cannot reach Ollama"**: start it with `ollama serve` (or open the Ollama app) and check `OLLAMA_URL`.
* **"model is not installed"**: run the `ollama pull ...` command shown in the message.
* **`pip install openai-whisper` fails**: it needs PyTorch; try Python 3.11/3.12 for the venv, or use `faster_whisper`.
* **Transcript is empty or full of "thanks for watching"**: no audio reached the app (enable *Share audio* in the
  browser's share dialog). Whisper invents text on silence; known phrases are filtered, and `faster_whisper`
  also skips silence.
* **Wrong language / garbled names**: set `WHISPER_LANGUAGE` and put names in `WHISPER_PROMPT`.
* **Report shows a ⚠ warning**: the analysis model failed on some sections (rate limit, timeout, model down).
  The message says why. Fix the cause, then rebuild the report with `curl -X POST http://127.0.0.1:8000/api/meetings/<id>/end` (list meeting ids with `curl http://127.0.0.1:8000/api/projects/<project id>/meetings`; the first project is id 1. The transcript is kept, so nothing is lost).
* **Groq rate limits**: requests are retried with backoff and rotate through multiple comma-separated keys.

## Known limitations

* An open question is judged within its slice plus a few lines after it. A question raised early and answered much
  later in the meeting can still be listed as open.
* All audio is one stream, so owners come from names spoken in the conversation, not voice identification.
* Chunked live capture can clip a word at a chunk boundary.
* Local Whisper and Ollama paths are covered by tests with stand-ins; real-model quality depends on the model and
  hardware you choose, which is what `scripts/eval_extraction.py` is for.

## Security

Keep `.env` private: it holds your API key and is excluded by `.gitignore`. Don't zip or share the project folder
with `.env` inside. If a key was ever shared, rotate it in the Groq console.
