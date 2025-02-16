import logging
import io
import os
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pyannote.audio import Pipeline
from app.utils.audio_utils import (
    convert_audio_format,
    validate_audio_format,
    merge_speaker_segments,
    extract_speaker_segments
)
from app.utils.config import get_huggingface_token
from app.utils.exceptions import InvalidAudioFormatError, AudioProcessingError, ModelLoadingError

# ✅ Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ✅ Load environment variables
AUDIO_SAVE_PATH = "saved_audio"
SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")  # Default to localhost if not set
AUDIO_URL_BASE = f"http://{SERVER_IP}:7000/audio"

# ✅ Ensure the folder for saving audio exists
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

# ✅ Load Hugging Face Token
HUGGINGFACE_TOKEN = get_huggingface_token()

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Fix CORS issues for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(AUDIO_SAVE_PATH, filename)

    # Check if file exists before serving
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    return FileResponse(file_path, media_type="audio/wav")

# ✅ Load the diarization pipeline
try:
    logging.info("⏳ Loading Pyannote Speaker Diarization Model...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
    logging.info("✅ Pyannote model loaded successfully!")
except Exception as e:
    logging.error(f"🚨 Failed to load Pyannote model: {e}")
    raise ModelLoadingError()

@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    """
    Process an uploaded audio file, segment it by speakers, and return results.
    """
    logging.info(f"📥 Received file: {file.filename}, Content-Type: {file.content_type}")

    # ✅ Read file into memory
    original_audio = await file.read()

    try:
        # ✅ Validate format before processing
        if not validate_audio_format(original_audio):
            original_audio = convert_audio_format(original_audio)  # Convert in-memory

        # ✅ Process the file using Pyannote
        logging.info("🔄 Processing audio for diarization...")
        diarization_result = pipeline(io.BytesIO(original_audio))
        logging.info("✅ Speaker diarization completed!")

        # ✅ Extract speaker segments
        raw_segments = [
            {"speaker": speaker, "start": round(segment.start, 2), "end": round(segment.end, 2)}
            for segment, _, speaker in diarization_result.itertracks(yield_label=True)
        ]

        # ✅ Merge speaker segments to remove small gaps
        merged_segments = merge_speaker_segments(raw_segments)

        # ✅ Extract and save speaker-specific audio clips
        speaker_audio_segments = extract_speaker_segments(original_audio, merged_segments, AUDIO_SAVE_PATH, AUDIO_URL_BASE)

        logging.info(f"📊 Processed {len(merged_segments)} merged segments from {file.filename}")

    except InvalidAudioFormatError as e:
        raise e
    except AudioProcessingError as e:
        raise e
    except Exception as e:
        logging.error(f"🚨 Error processing audio: {e}")
        raise AudioProcessingError("Unexpected error occurred during processing.")

    # ✅ Return structured response with file URLs
    return {
        "file": file.filename,
        "speakers": speaker_audio_segments
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure API and diarization model are running correctly.
    """
    try:
        # ✅ Check if the diarization model is loaded
        if not pipeline:
            raise RuntimeError("Diarization model is not loaded.")

        return {"status": "ok", "model": "loaded"}

    except Exception as e:
        logging.error(f"🚨 Health check failed: {e}")
        return {"status": "error", "message": str(e)}

# ✅ Start Uvicorn automatically if running standalone
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000, reload=True)
