import requests
import base64
import time
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG ================= #

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
POLL_INTERVAL = 10  # seconds

RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

# ========================================== #

def is_url(path: str):
    try:
        return urlparse(path).scheme in ("http", "https")
    except:
        return False


def load_text(text_source: str) -> str:
    """
    text_source = file path OR url
    """
    if is_url(text_source):
        print(f"[+] Downloading text from URL: {text_source}")
        r = requests.get(text_source)
        r.raise_for_status()
        return r.text.strip()
    else:
        print(f"[+] Reading text from file: {text_source}")
        with open(text_source, "r", encoding="utf-8") as f:
            return f.read().strip()


def load_audio_base64(audio_source: str) -> str:
    """
    audio_source = file path OR url
    """
    if is_url(audio_source):
        print(f"[+] Downloading audio from URL: {audio_source}")
        audio_bytes = requests.get(audio_source).content
    else:
        print(f"[+] Reading audio from file: {audio_source}")
        with open(audio_source, "rb") as f:
            audio_bytes = f.read()

    return base64.b64encode(audio_bytes).decode("utf-8")


def submit_job(text: str, base64_audio: str):
    payload = {
        "input": {
            "text": text,
            "ref_audio": base64_audio
        }
    }

    print("[+] Submitting job to RunPod...")

    r = requests.post(RUN_URL, headers=HEADERS, json=payload)
    r.raise_for_status()
    res = r.json()

    job_id = res.get("id")
    print(f"[+] Job submitted: {job_id}")
    return job_id


def poll_job(job_id: str):
    print("[+] Polling job status every 60 seconds...")

    while True:
        r = requests.get(f"{STATUS_URL}/{job_id}", headers=HEADERS)
        r.raise_for_status()
        res = r.json()

        status = res.get("status")
        print(f"[{time.strftime('%H:%M:%S')}] Status:", status)

        if status == "COMPLETED":
            print("[+] Job completed!")
            return res

        if status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
            raise Exception(f"Job failed with status: {status}")

        time.sleep(POLL_INTERVAL)


def save_output_audio(result: dict, out_file="output.wav"):
    output = result.get("output", {})

    b64_audio = output.get("audio")
    sr = output.get("sr", 22050)
    fmt = output.get("format", "wav")

    if not b64_audio:
        raise Exception("No audio in output!")

    audio_bytes = base64.b64decode(b64_audio)

    out_file = f"output.{fmt}"

    with open(out_file, "wb") as f:
        f.write(audio_bytes)

    print(f"[+] Audio saved as: {out_file}")
    print(f"[+] Sample rate: {sr}")


# ================= MAIN ================= #

if __name__ == "__main__":

    # -------- INPUTS -------- #
    TEXT_SOURCE = "test_res.txt"  
    # or URL:
    # TEXT_SOURCE = "https://example.com/input.txt"

    AUDIO_SOURCE = "ref.wav"
    # or URL:
    # AUDIO_SOURCE = "https://files.1185912.xyz/api/public/dl/I9w7a1lh"

    # ------------------------ #

    text = load_text(TEXT_SOURCE)
    base64_audio = load_audio_base64(AUDIO_SOURCE)

    job_id = submit_job(text, base64_audio)
    result = poll_job(job_id)

    save_output_audio(result)
