import time, uuid
import torch, librosa, soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from qwen_tts import Qwen3TTSModel

app = FastAPI()



device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Qwen3-TTS model...")
t0 = time.time()

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
print("Model loaded in", round(time.time()-t0, 2), "seconds")

@app.post("/voice-clone")
async def voice_clone(
    text: str = Form(...),
    emotion: str = Form(None),
    ref_audio: UploadFile = File(...)
):
    start_time = time.time()

    ref_path = f"/tmp/{uuid.uuid4()}.wav"
    out_path = f"/tmp/{uuid.uuid4()}.wav"

    with open(ref_path, "wb") as f:
        f.write(await ref_audio.read())

    ref_audio_data, sr = librosa.load(ref_path, sr=None, mono=True)

    # Chunking logic
    from utils import smart_split_text
    import numpy as np
    
    text_chunks = smart_split_text(text, max_length=500)
    all_audio = []
    final_sr = None
    
    print(f"Processing {len(text_chunks)} chunks...")
    
    for i, chunk in enumerate(text_chunks):
        input_text = chunk
        if emotion:
            input_text = f"[Emotion: {emotion}] " + chunk
            
        wavs, out_sr = model.generate_voice_clone(
            text=input_text,
            ref_audio=(ref_audio_data, sr),
            ref_text=None,
            x_vector_only_mode=True
        )
        
        if wavs:
            all_audio.append(wavs[0])
            final_sr = out_sr

    if all_audio:
        stitched_audio = np.concatenate(all_audio)
        sf.write(out_path, stitched_audio, final_sr)
    else:
        # Fallback or error handling
        print("No audio generated")
        return {"error": "Generation failed"}

    gen_time = round(time.time() - start_time, 2)
    print("Generated in", gen_time, "seconds")

    return FileResponse(out_path, media_type="audio/wav", filename="clone.wav")



# Serve UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if  __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="[IP_ADDRESS]", port=8000)