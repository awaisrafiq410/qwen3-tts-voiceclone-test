import time, uuid
import torch, librosa, soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

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

    if emotion:
        text = f"[Emotion: {emotion}] " + text

    wavs, out_sr = model.generate_voice_clone(
        text=text,
        ref_audio=(ref_audio_data, sr),
        ref_text=None,
        x_vector_only_mode=True
    )

    sf.write(out_path, wavs[0], out_sr)

    gen_time = round(time.time() - start_time, 2)
    print("Generated in", gen_time, "seconds")

    return FileResponse(out_path, media_type="audio/wav", filename="clone.wav")



# Serve UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if  __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="[IP_ADDRESS]", port=8000)