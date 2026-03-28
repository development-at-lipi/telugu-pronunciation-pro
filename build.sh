#!/usr/bin/env bash
# Render build script — installs Python deps + ffmpeg (needed for Google Cloud fallback)
set -e

pip install -r requirements.txt

# ffmpeg needed only for Google Cloud STT (converts WebM → WAV)
# Sarvam AI accepts WebM directly — no ffmpeg needed for primary engine
apt-get update -qq && apt-get install -y -qq ffmpeg || echo "ffmpeg install skipped"
