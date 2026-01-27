import runpod
import torch
import soundfile as sf
import librosa
import numpy as np
import base64
import io
import time
from qwen_tts import Qwen3TTSModel

# --- 1. Global Model Initialization (Cold Start) ---
print("Initializing RunPod Handler...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print("Loading model...")
    t0 = time.time()
    # We expect weights to be baked into the image or cached in the default HF cache
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    print(f"Model loaded in {round(time.time()-t0, 2)}s")
except Exception as e:
    print(f"FAILED to load model: {e}")
    model = None

# --- 2. Helper Functions ---
def decode_audio(base64_str):
    audio_bytes = base64.b64decode(base64_str)
    audio_buffer = io.BytesIO(audio_bytes)
    # librosa.load supports file-like objects
    data, sr = librosa.load(audio_buffer, sr=None, mono=True)
    return data, sr

def encode_audio(audio_data, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format='WAV')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

# --- 3. Handler Function ---
async def handler(job):
    job_input = job["input"]
    
    # Validation
    if not model:
        return {"error": "Model not loaded properly."}
    
    text = job_input.get("text")
    if not text:
        return {"error": "Missing 'text' in input."}
        
    ref_audio_b64 = job_input.get("ref_audio")
    if not ref_audio_b64:
        # Check if ref_audio is a URL (implementation complexity: stick to b64 for now as per plan)
        return {"error": "Missing 'ref_audio' (base64 string) in input."}
        
    emotion = job_input.get("emotion") # Optional

    print(f"Processing job {job['id']}: text='{text[:20]}...', emotion='{emotion}'")

    try:
        # Decode Ref Audio
        ref_audio_data, sr = decode_audio(ref_audio_b64)

        # Preprocess Text
        if emotion:
            text = f"[Emotion: {emotion}] " + text

        # Inference
        t_infer = time.time()
        wavs, out_sr = model.generate_voice_clone(
            text=text,
            ref_audio=(ref_audio_data, sr),
            ref_text=None,
            x_vector_only_mode=True
        )
        print(f"Inference time: {round(time.time()-t_infer, 2)}s")

        # Encode Output
        if not wavs:
            return {"error": "No audio generated."}
            
        b64_output = encode_audio(wavs[0], out_sr)

        return {
            "audio": b64_output, 
            "sr": out_sr,
            "format": "wav"
        }

    except Exception as e:
        print(f"Error processing job: {e}")
        return {"error": str(e)}

# --- 4. Start RunPod ---
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
