"""FFmpeg helpers."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

INSTALL_HINT = 'FFmpeg is not installed. Fedora: sudo dnf install ffmpeg  ·  Ubuntu: sudo apt install ffmpeg  ·  macOS: brew install ffmpeg'


def ffmpeg_path() -> str:
    ff = shutil.which('ffmpeg')
    if not ff:
        raise RuntimeError(INSTALL_HINT)
    return ff


def extract_wav(src: Path) -> Path:
    """Decode browser WebM/Opus (or any audio) to 16 kHz mono WAV."""
    out = src.with_name(f'{src.stem}_16k.wav')  # never the input path itself (a .wav upload would collide)
    cmd = [ffmpeg_path(), '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src),
           '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if p.returncode or not out.exists() or out.stat().st_size < 1000:
        raise RuntimeError('FFmpeg could not decode this audio. The browser may have captured an empty stream. ' + p.stderr[-800:])
    return out


def run_ff(args, timeout=3600):
    p = subprocess.run([ffmpeg_path(), '-hide_banner', '-loglevel', 'error', '-y', *args],
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError('FFmpeg failed: ' + p.stderr[-600:])


def ffprobe_duration(p: Path) -> float:
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(p)],
                             capture_output=True, text=True, timeout=120).stdout.strip()
        return float(out)
    except Exception:
        return 0.0
