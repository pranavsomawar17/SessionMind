#!/usr/bin/env bash
# ./run.sh            start SessionMind
# ./run.sh --local    also install the local Whisper packages (requirements-local.txt)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  python3 -m venv venv || { echo "Could not create a virtual environment. On Ubuntu/Debian: sudo apt install python3-venv"; exit 1; }
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q -r requirements.txt
if [ "${1:-}" = "--local" ]; then pip install -r requirements-local.txt; fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Open it and choose your engines (cloud or local)."
fi
command -v ffmpeg >/dev/null 2>&1 || echo "Warning: ffmpeg is not installed. Fedora: sudo dnf install ffmpeg | Ubuntu: sudo apt install ffmpeg | macOS: brew install ffmpeg"

echo "SessionMind: http://127.0.0.1:8000"
exec python -m uvicorn app.main:app --reload --reload-dir app --reload-dir static
