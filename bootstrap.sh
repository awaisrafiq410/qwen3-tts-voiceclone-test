#!/bin/bash
set -e

echo "=== Qwen3-TTS Voice Clone Bootstrap ==="

apt update && apt install -y git ffmpeg python3 python3-pip python3-venv curl

git clone https://github.com/awaisrafiq410/qwen3-tts-voiceclone-test.git
cd qwen3-tts-voiceclone-test

chmod +x setup.sh run.sh test.sh

./setup.sh
./run.sh
