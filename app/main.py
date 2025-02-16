import logging
import base64
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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

        # ✅ Extract speaker-specific audio clips
        speaker_audio_segments = extract_speaker_segments(original_audio, merged_segments)

        logging.info(f"📊 Processed {len(merged_segments)} merged segments from {file.filename}")

    except InvalidAudioFormatError as e:
        raise e
    except AudioProcessingError as e:
        raise e
    except Exception as e:
        logging.error(f"🚨 Error processing audio: {e}")
        raise AudioProcessingError("Unexpected error occurred during processing.")

    # ✅ Encode original audio to Base64 to prevent JSON errors
    original_audio_base64 = base64.b64encode(original_audio).decode("utf-8")

  # ✅ Encode speaker segments into Base64 (Ensure `segment["audio"]` is in bytes)
    for speaker in speaker_audio_segments:
        for segment in speaker_audio_segments[speaker]:
            if isinstance(segment["audio"], str):
                segment["audio"] = base64.b64encode(segment["audio"].encode("utf-8")).decode("utf-8")
            else:
                segment["audio"] = base64.b64encode(segment["audio"]).decode("utf-8")

    # ✅ Return structured response with encoded audio
    return {
        "file": file.filename,
        "original_audio": original_audio_base64,
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
