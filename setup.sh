#!/bin/bash
set -e

echo "=== System packages ==="
apt update && apt install -y git ffmpeg python3 python3-pip python3-venv curl

echo "=== Python venv ==="
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip

echo "=== Install Python deps ==="
pip install -r requirements.txt

echo "=== Clone Qwen3-TTS ==="
if [ ! -d "Qwen3-TTS-Openai-Fastapi" ]; then
  git clone https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git
fi

cd Qwen3-TTS-Openai-Fastapi
pip install -e ".[api]"
pip install -U flash-attn --no-build-isolation
cd ..

echo "=== Setup complete ==="
