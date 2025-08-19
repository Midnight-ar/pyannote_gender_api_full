import os
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline
from inaSpeechSegmenter import Segmenter

HF_TOKEN = os.getenv("HF_TOKEN")

# Load pyannote pipeline (speaker diarization)
pipeline = None
if HF_TOKEN:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

# inaSpeechSegmenter (gender detection)
segmenter = Segmenter()

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), min_segment: float = Form(0.7), engine: str = Form("auto")):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        results = []

        if engine in ["auto", "ina"]:
            segs = segmenter(tmp_path)
            for label, start, end in segs:
                if label in ["male", "female"] and (end - start) >= min_segment:
                    results.append({"gender": label, "start": float(start), "end": float(end)})

        if engine in ["auto", "pyannote"] and pipeline:
            diarization = pipeline(tmp_path)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                results.append({"speaker": speaker, "start": turn.start, "end": turn.end})

        return JSONResponse(content={"segments": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Gender Detection API running"}
