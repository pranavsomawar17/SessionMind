#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -r requirements.txt
if [ ! -f .env ]; then cp .env.example .env; fi
echo "SessionMind: http://127.0.0.1:8000"
exec python -m uvicorn app.main:app --reload
