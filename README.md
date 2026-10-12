# SessionMind — AI Meeting Intelligence MVP

A lightweight, real meeting capture MVP focused on **AI understanding and the final meeting report**, not manual transcript entry.

## Architecture

- **Browser:** `getDisplayMedia()` captures a permitted meeting window/tab and system audio.
- **FFmpeg:** converts the browser's WebM/Opus audio to 16 kHz mono WAV.
- **Groq Whisper:** speech-to-text, uploaded in ~10s chunks while the meeting is live.
- **Groq vision:** best-effort reading of live captions/slide text out of periodic screenshots.
- **Groq LLM:** runs incremental live task/decision/question extraction during the meeting, and analyzes the complete transcript + project history for the final report.
- **SQLite:** stores meetings, transcripts, screenshots, live-detected events, and reports.
- **python-docx:** creates the final Word (.docx) report, including a topic tree and an open-questions table.
- **HTML:** creates a lightweight browsable report.

This does not pretend to have private Google Meet/Teams/Zoom internal feeds. The user explicitly grants browser capture permission and must enable audio sharing.

## Requirements

- Python 3.10+
- Google Chrome or Chromium
- FFmpeg
- Groq API key

### Fedora

```bash
sudo dnf install ffmpeg
```

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_key
GROQ_TRANSCRIPTION_MODEL=whisper-large-v3-turbo
GROQ_CHAT_MODEL=llama-3.3-70b-versatile
```

No OpenAI key is required or used by this MVP.

## Run

```bash
source venv/bin/activate
python -m uvicorn app.main:app --reload
```

Open:

`http://127.0.0.1:8000`

## How to use

1. Select/create a project.
2. Enter a meeting title, and optionally a **speaker label** — who's about to talk (you can change this mid-meeting before each turn; there's no automatic diarization, so this is how speakers get named in the transcript).
3. Click **Start AI Meeting**.
4. Click **Share Meeting Screen + Audio**.
5. In the browser chooser, select the meeting tab/window.
6. Enable **Share audio**.
7. Conduct the meeting normally. The transcript now streams in roughly every 10 seconds instead of only appearing at the end.
8. Watch the **Live insights** panel: every ~20 seconds it re-checks the new transcript for tasks, decisions, and open questions, and shows the current topic. Confirm or dismiss anything it detects.
9. SessionMind also records periodic screenshots and (when a Groq vision model is available on your account) reads any live captions or slide text out of them, folding that into the transcript too.
10. Click **End & Generate Report**.
11. The complete transcript is sent to Groq LLM for structured meeting understanding.
12. Download the Word (.docx) report, or open the HTML version.

## Report contents

The report can contain:

- Executive summary
- Meeting overview
- Participants (only when supported by transcript)
- Key discussion points
- Confirmed decisions with evidence
- Action-item table with owner/deadline/status
- Open questions (full-transcript scan plus live detection; every unresolved item is kept, with status and evidence)
- Risks/blockers
- Topic tree / mind map: root meeting → themes → key points, decisions, tasks, risks and open questions
- Important moments
- Changes from previous project meetings
- Meeting screenshots
- Evidence transcript

The AI is instructed not to invent decisions, owners, deadlines or progress.

## What changed in this version

- **Fixed a bug where the transcript never appeared until the meeting ended.** `MediaRecorder.start()` was called with no timeslice, so audio only ever flushed once, at `stop()` — for a long meeting this also risked hitting Groq Whisper's file-size limit in one giant upload. Live audio is now captured as a sequence of independently valid WebM files, one every 10 seconds (see next point for why that's not just `start(10000)`), so the "Live session" panel actually updates live and each upload stays small.
- **Correction to that fix:** the first pass used `MediaRecorder.start(10000)` (a timeslice), which produces a stream of chunks that are *not* independently decodable — only the very first chunk has a WebM/EBML header; every chunk after that is a headerless continuation fragment, so ffmpeg correctly rejects it with `EBML header parsing failed`. The fix instead **stops and restarts** the recorder every 10 seconds on the same audio stream. Each `stop()` flushes one complete, self-contained WebM file, so every uploaded chunk decodes cleanly. The trade-off is a sub-second gap in the recording at each rotation point — negligible for meeting transcription, but worth knowing about.
- **Live task/decision/question detection.** The `events` table existed in the schema but nothing ever wrote to it. There's now a `/api/meetings/{id}/analyze` endpoint, polled every 20s while capturing, that reads new transcript since the last check and extracts tasks/decisions/questions/topic — shown in a new **Live insights** panel with confirm/dismiss controls, per the original spec's "Live Dashboard" requirement.
- **Screenshots are now actually analyzed.** They were captured and stored but never read. A best-effort Groq vision call now reads live captions or slide/document text out of each one and folds it into the transcript — this is what makes the report meaningfully reflect what was *shown*, not just said. If your account has no vision-capable model available, this silently no-ops rather than breaking capture.
- **Manual speaker labeling**, since there's no real diarization — a "Speaker label" field tags who's talking, sent with each audio chunk.
- **Fixed an event-loop-blocking bug**: the audio and screenshot endpoints were `async def` but called blocking code (`ffmpeg`, `requests`) directly inside them, which stalls every other in-flight request for the duration of each transcription call. They're plain `def` now, so FastAPI runs them in a worker thread instead.
- **Added `PRAGMA busy_timeout`** so the now much more frequent SQLite writes (chunked audio + screenshots + live events, all roughly every 10–20s) don't intermittently fail with "database is locked" under load.
- Fixed a duplicated line in the requirements note in this README.

## Important capture note

On Linux, browser screen/audio sharing support varies by browser and capture source. If **Share audio** is unavailable for a particular window, try sharing the **meeting browser tab** instead. The meeting tab must expose its audio to the browser's screen-capture API.

## Troubleshooting

### `GROQ_API_KEY` missing

Run:

```bash
cat .env
```

Make sure the key is in `GROQ_API_KEY`, not `OPENAI_API_KEY`.

### FFmpeg error

```bash
ffmpeg -version
```

If unavailable on Fedora:

```bash
sudo dnf install ffmpeg
```

### No audio captured

Restart capture and choose the meeting tab/window. Enable **Share audio**. A screenshot can work without audio; transcription cannot.

### No transcript

The meeting must contain audible speech and the capture dialog must share system audio. Very tiny recordings are ignored.

### Live insights panel stays empty

This calls the same Groq chat model as the final report, just more often and on a smaller window of text. If it's empty, check the browser console/network tab for `/api/meetings/{id}/analyze` errors — the most likely cause is the same as any other Groq call failing (bad key, rate limit).

## Security

Keep `.env` private. API keys are used only by the backend and are not sent to the browser UI.

