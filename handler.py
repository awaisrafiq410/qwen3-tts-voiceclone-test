import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import runpod
import torch
import soundfile as sf
import librosa
import numpy as np
import base64
import io
import time
from qwen_tts import Qwen3TTSModel

print("Initializing RunPod Handler...")

# --- HARD GPU CHECK ---
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available in this worker")

# --- CLEAN CUDA STATE ---
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("CUDA device:", torch.cuda.get_device_name(0))

# --- MODEL LOAD (FIXED) ---
try:
    print("Loading model...")
    t0 = time.time()

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map={"": "cuda:0"},   # 👈 THIS IS CRITICAL
        torch_dtype=torch.float16
    )

    print(f"Model loaded in {round(time.time() - t0, 2)}s")

except Exception as e:
    print("FAILED to load model:", e)
    model = None

# --- HELPERS ---
def decode_audio(base64_str):
    audio_bytes = base64.b64decode(base64_str)
    audio_buffer = io.BytesIO(audio_bytes)
    return librosa.load(audio_buffer, sr=None, mono=True)

def encode_audio(audio_data, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# --- HANDLER ---
async def handler(job):
    start = time.time()

    if model is None:
        return {"error": "Model not loaded"}

    inp = job["input"]
    text = inp.get("text")
    ref_audio_b64 = inp.get("ref_audio")
    emotion = inp.get("emotion")

    if not text or not ref_audio_b64:
        return {"error": "Missing text or ref_audio"}

    # Decode audio
    ref_audio, sr = decode_audio(ref_audio_b64)

    # Split text
    from utils import smart_split_text
    chunks = smart_split_text(text, max_length=800)  # larger chunks = faster

    audio_parts = []
    final_sr = None

    for chunk in chunks:
        if emotion:
            chunk = f"[Emotion: {emotion}] " + chunk

        wavs, out_sr = model.generate_voice_clone(
            text=chunk,
            ref_audio=(ref_audio, sr),
            ref_text=None,
            x_vector_only_mode=True
        )

        audio_parts.append(wavs[0])
        final_sr = out_sr

    stitched = np.concatenate(audio_parts)
    b64_out = encode_audio(stitched, final_sr)

    return {
        "audio": b64_out,
        "sr": final_sr,
        "format": "wav",
        "time": round(time.time() - start, 2),
        "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    }

# --- START ---
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
