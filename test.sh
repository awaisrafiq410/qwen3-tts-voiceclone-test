#!/bin/bash
RUNPOD_IP=$1

if [ -z "$RUNPOD_IP" ]; then
  echo "Usage: ./test.sh <RUNPOD_IP>"
  exit 1
fi

curl -X POST http://$RUNPOD_IP:8000/voice-clone \
-F "text=Automated voice cloning test using Qwen3 TTS" \
-F "emotion=Calm" \
-F "ref_audio=@ref.wav"
