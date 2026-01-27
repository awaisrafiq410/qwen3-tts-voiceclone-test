#!/bin/bash
set -e

echo "=== System packages ==="
apt update && apt install -y git ffmpeg python3 python3-pip python3-venv curl

echo "=== Python venv ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "=== Install Python deps ==="
pip install -r requirements.txt

echo "=== Clone Qwen3-TTS ==="
if [ ! -d "Qwen3-TTS" ]; then
  git clone https://github.com/QwenLM/Qwen3-TTS.git
fi

cd Qwen3-TTS
pip install -e .
cd ..

echo "=== Setup complete ==="
