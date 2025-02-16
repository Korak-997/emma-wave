import os
import logging
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pyannote.audio import Pipeline
from app.utils.audio_utils import convert_audio_format, validate_audio_format
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
    allow_origins=["*"],  # Change to specific domains later for security
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
    Process an uploaded audio file and return speaker diarization results.
    """
    logging.info(f"📥 Received file: {file.filename}")

    # ✅ Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file.file.read())

    try:
        # ✅ Validate format before processing
        if not validate_audio_format(temp_filename):
            temp_filename = convert_audio_format(temp_filename)

        # ✅ Process the file using Pyannote
        logging.info("🔄 Processing audio for diarization...")
        diarization_result = pipeline(temp_filename)
        logging.info("✅ Speaker diarization completed!")

        # ✅ Extract speaker segments
        segments = [
            {"speaker": speaker, "start": round(segment.start, 2), "end": round(segment.end, 2)}
            for segment, _, speaker in diarization_result.itertracks(yield_label=True)
        ]
        logging.info(f"📊 Processed {len(segments)} segments from {file.filename}")

    except InvalidAudioFormatError as e:
        raise e  # Return directly since it's a 400 error.
    except AudioProcessingError as e:
        raise e
    except Exception as e:
        logging.error(f"🚨 Error processing audio: {e}")
        raise AudioProcessingError("Unexpected error occurred during processing.")
    finally:
        os.remove(temp_filename)
        logging.info(f"🗑️ Deleted temp file: {temp_filename}")

    return {"file": file.filename, "segments": segments}
