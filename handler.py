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
print("Device:", device)

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
    t0 = time.time()
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

        # Preprocess Text and split
        from utils import smart_split_text
        text_chunks = smart_split_text(text, max_length=500)
        
        all_generated_audio = []
        final_sr = None

        print(f"Splitting text into {len(text_chunks)} chunks...")

        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue

            # Prepend emotion to EACH chunk if specified, to maintain consistency
            input_text = chunk_text
            if emotion:
                input_text = f"[Emotion: {emotion}] " + chunk_text
            
            print(f"Processing chunk {i+1}/{len(text_chunks)}: {input_text[:30]}...")

            # Inference
            wavs, out_sr = model.generate_voice_clone(
                text=input_text,
                ref_audio=(ref_audio_data, sr),
                ref_text=None,
                x_vector_only_mode=True
            )
            
            if wavs and len(wavs) > 0:
                all_generated_audio.append(wavs[0])
                final_sr = out_sr

        # Stitch
        if not all_generated_audio:
             return {"error": "No audio generated from chunks."}

        stitched_audio = np.concatenate(all_generated_audio)
        
        # Encode Output
        b64_output = encode_audio(stitched_audio, final_sr)

        # save stitched audio to file
        sf.write(f"output_{job['id']}.wav", stitched_audio, final_sr)

        t_end = time.time()
        print(f"Job {job['id']} completed in {round(t_end-t0, 2)}s")

        return {
            "audio": b64_output, 
            "sr": final_sr,
            "format": "wav",
            "device": device,
            "time": round(t_end-t0, 2)
        }

    except Exception as e:
        print(f"Error processing job: {e}")
        return {"error": str(e)}

# --- 4. Start RunPod ---
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
