FROM python:3.11-slim

# System deps: ffmpeg needed by inaSpeechSegmenter
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py ./

# Env: set at runtime
# ENV HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
