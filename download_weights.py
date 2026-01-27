# download_weights.py
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
import torch

print("Downloading and caching model weights...")
Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cpu", # Valid download even without GPU
    torch_dtype=torch.float32
)
print("Model weights downloaded successfully.")
