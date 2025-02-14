import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
import shutil
from pyannote.audio import Pipeline

# Load environment variables from .env
load_dotenv()

# Get the Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")

if not HUGGINGFACE_TOKEN:
    raise ValueError("🚨 HUGGINGFACE_TOKEN is not set! Make sure it's in your .env file.")

print(f"✅ Using Hugging Face Token: {HUGGINGFACE_TOKEN[:10]}... (truncated)")

app = FastAPI()

# Load the diarization pipeline with the token
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HUGGINGFACE_TOKEN
)

@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    diarization_result = pipeline(temp_filename)

    segments = []
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": segment.start,
            "end": segment.end
        })

    return {"segments": segments}
