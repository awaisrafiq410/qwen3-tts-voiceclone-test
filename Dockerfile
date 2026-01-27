# Stage 1: Fetch static ffmpeg
FROM mwader/static-ffmpeg:6.0 AS ffmpeg

# Stage 2: Final Image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Copy ffmpeg
COPY --from=ffmpeg /ffmpeg /usr/local/bin/
COPY --from=ffmpeg /ffprobe /usr/local/bin/

WORKDIR /app

# Install system dependencies
# libsndfile1 is required for soundfile/librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir scipy git+https://github.com/huggingface/transformers

COPY handler.py .
COPY download_weights.py .

# Bake weights into the image
# This increases build time and image size but drastically reduces cold start time
RUN python download_weights.py

# Clean up download script
RUN rm download_weights.py

# Start the handler
CMD ["python", "-u", "handler.py"]
