# ---------------- ENV (MUST be first) ----------------
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------- Imports ----------------
import runpod
import torch
import soundfile as sf
import numpy as np
import base64
import io
import time
import uuid
import boto3
from qwen_tts import Qwen3TTSModel
import soundfile as sf

print("Initializing RunPod Handler...")

CHUNK_SIZE = os.getenv("CHUNK_SIZE", 1000)

# ---------------- CUDA CHECK ----------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("CUDA device:", torch.cuda.get_device_name(0))

# ---------------- MODEL LOAD ----------------
model = None
try:
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16
    )
    print("Model loaded in", round(time.time() - t0, 2), "s")
except Exception as e:
    print("MODEL LOAD FAILED:", e)
    print(torch.cuda.memory_summary())
    model = None

# ---------------- HELPERS ----------------
def decode_audio(b64):
    audio_bytes = base64.b64decode(b64)
    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr

def encode_audio(audio, sr):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def save_to_r2(audio_data, sr):
    print("Saving to Cloudflare R2...")

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")
    subfolder = os.environ.get("R2_SUBFOLDER", "")

    if not all([account_id, access_key, secret_key, bucket_name]):
        raise ValueError("Missing R2 configuration environment variables")

    filename = f"{uuid.uuid4()}.wav"
    key = f"{subfolder.strip('/')}/{filename}" if subfolder else filename

    # Write WAV to memory
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format="WAV")
    buffer.seek(0)

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    s3.upload_fileobj(buffer, bucket_name, key)

    print("R2 upload complete:", key)
    return key

# ---------------- HANDLER ----------------
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

    # Decode reference audio
    ref_audio, sr = decode_audio(ref_audio_b64)

    # Split text (SAFE SIZE)
    from utils import smart_split_text
    chunks = smart_split_text(text, max_length=CHUNK_SIZE)

    audio_parts = []
    final_sr = None

    with torch.inference_mode():
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

    # Encode
    b64_out = encode_audio(stitched, final_sr)

    # Optional R2 upload (non-blocking safety)
    r2_key = None
    try:
        if os.environ.get("R2_BUCKET_NAME"):
            r2_key = save_to_r2(stitched, final_sr)
    except Exception as e:
        print("R2 upload failed:", e)

    # ---- cleanup for warm worker ----
    del ref_audio
    del audio_parts
    torch.cuda.empty_cache()

    return {
        "audio": b64_out,
        "sr": final_sr,
        "format": "wav",
        "r2_key": r2_key,
        "time": round(time.time() - start, 2),
        "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    }

# ---------------- START ----------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
