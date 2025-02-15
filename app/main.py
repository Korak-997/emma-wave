import os
import shutil
import logging
import subprocess
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from pyannote.audio import Pipeline

# ✅ Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Get the Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")

if not HUGGINGFACE_TOKEN:
    logging.error("🚨 HUGGINGFACE_TOKEN is missing! Make sure it's set in your .env file.")
    raise ValueError("HUGGINGFACE_TOKEN is not set!")

logging.info(f"✅ Using Hugging Face Token: {HUGGINGFACE_TOKEN[:10]}... (truncated)")

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Load the diarization pipeline
try:
    logging.info("⏳ Loading Pyannote Speaker Diarization Model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    logging.info("✅ Pyannote model loaded successfully!")
except Exception as e:
    logging.error(f"🚨 Failed to load Pyannote model: {e}")
    raise RuntimeError("Failed to load diarization model!")

# ✅ Function to convert audio to required format
def convert_audio_format(input_file):
    output_file = f"converted_{input_file}"
    command = [
        "ffmpeg", "-i", input_file,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_file
    ]
    try:
        subprocess.run(command, check=True)
        logging.info(f"✅ Audio successfully converted: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logging.error(f"🚨 Audio conversion failed: {e}")
        raise HTTPException(status_code=500, detail="Audio format conversion failed.")

# ✅ API Endpoint for Diarization
@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    """
    Process an uploaded audio file and return speaker diarization results.
    """
    logging.info(f"📥 Received file: {file.filename}")

    # ✅ Save the uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"✅ File saved successfully: {temp_filename}")
    except Exception as e:
        logging.error(f"🚨 Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Error saving file.")

    # ✅ Convert audio file to required format
    try:
        converted_filename = convert_audio_format(temp_filename)
    except HTTPException:
        raise HTTPException(status_code=500, detail="Failed to convert audio format.")

    # ✅ Process the file using Pyannote
    try:
        logging.info("🔄 Processing audio for diarization...")
        diarization_result = pipeline(converted_filename)
        logging.info("✅ Speaker diarization completed!")
    except Exception as e:
        logging.error(f"🚨 Diarization failed: {e}")
        raise HTTPException(status_code=500, detail="Diarization failed.")

    # ✅ Extract speaker segments
    segments = []
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2)
        })

    logging.info(f"📊 Processed {len(segments)} segments from {file.filename}")

    # ✅ Clean up the temporary files
    try:
        os.remove(temp_filename)
        os.remove(converted_filename)
        logging.info(f"🗑️ Deleted temp files: {temp_filename}, {converted_filename}")
    except Exception as e:
        logging.warning(f"⚠️ Could not delete temp files: {e}")

    return {"file": file.filename, "segments": segments}
