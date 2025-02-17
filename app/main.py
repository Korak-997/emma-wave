import logging
import io
import os
import uuid
import time
import datetime
import traceback
import psutil
import asyncio  # Use asyncio for better performance
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pyannote.audio import Pipeline
from app.utils.audio_utils import (
    convert_audio_format,
    validate_audio_format,
    merge_speaker_segments,
    extract_speaker_segments
)
from app.utils.config import get_huggingface_token
from app.utils.exceptions import InvalidAudioFormatError, AudioProcessingError, ModelLoadingError
from app.utils.logging_utils import save_request_log, get_system_metrics

# ✅ Load environment variables
AUDIO_SAVE_PATH = "saved_audio"
SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")  # Default to localhost if not set
AUDIO_URL_BASE = f"http://{SERVER_IP}:7000/audio"
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"  # Control logging

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
    Tracks performance and system resource usage if logging is enabled.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    initial_metrics = get_system_metrics() if ENABLE_LOGGING else {}

    logging.info(f"📥 Received file: {file.filename}, Content-Type: {file.content_type}")

    # ✅ Read file into memory
    original_audio = await file.read()
    file_size = len(original_audio)  # Get size of uploaded file

    try:
        step_1_start = time.time()
        # ✅ Validate format before processing
        if not validate_audio_format(original_audio):
            original_audio = convert_audio_format(original_audio)  # Convert in-memory
        step_1_end = time.time()

        step_2_start = time.time()
        during_processing_metrics = []

        # ✅ Start diarization process
        logging.info("🔄 Processing audio for diarization...")

        if ENABLE_LOGGING:
            async def collect_metrics():
                while time.time() - step_2_start < 5:  # Collect every 5 seconds
                    await asyncio.sleep(0.1)  # Non-blocking alternative to time.sleep
                    during_processing_metrics.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "system": get_system_metrics(),
                    })
            asyncio.create_task(collect_metrics())  # Run metrics collection in the background

        diarization_result = pipeline(io.BytesIO(original_audio))
        logging.info("✅ Speaker diarization completed!")
        step_2_end = time.time()

        step_3_start = time.time()
        # ✅ Extract speaker segments
        raw_segments = [
            {"speaker": speaker, "start": round(segment.start, 2), "end": round(segment.end, 2)}
            for segment, _, speaker in diarization_result.itertracks(yield_label=True)
        ]

        # ✅ Merge speaker segments to remove small gaps
        merged_segments = merge_speaker_segments(raw_segments)

        # ✅ Extract and save speaker-specific audio clips
        speaker_audio_segments = extract_speaker_segments(original_audio, merged_segments, AUDIO_SAVE_PATH, AUDIO_URL_BASE)
        step_3_end = time.time()

        logging.info(f"📊 Processed {len(merged_segments)} merged segments from {file.filename}")

    except (InvalidAudioFormatError, AudioProcessingError) as e:
        logging.error(f"🚨 Processing Error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"🚨 Unexpected Error: {e}\n{error_trace}")

        if ENABLE_LOGGING:
            error_log = {
                "request_id": request_id,
                "file": file.filename,
                "error_message": str(e),
                "traceback": error_trace,
                "system_metrics_at_failure": get_system_metrics()
            }
            await save_request_log(error_log)

        return JSONResponse(status_code=500, content={"error": "Unexpected error occurred during processing."})

    # ✅ Capture final performance metrics
    final_metrics = get_system_metrics() if ENABLE_LOGGING else {}
    total_time = time.time() - start_time

    # ✅ Save performance log only if logging is enabled
    if ENABLE_LOGGING:
        performance_data = {
            "request_id": request_id,
            "file": file.filename,
            "file_size_bytes": file_size,
            "completion_timestamp": datetime.datetime.now().isoformat(),
            "processing_time_seconds": total_time,
            "step_timings": {
                "audio_conversion": step_1_end - step_1_start,
                "diarization_processing": step_2_end - step_2_start,
                "segment_extraction": step_3_end - step_3_start,
            },
            "system_metrics": {
                "before_processing": initial_metrics,
                "during_processing": during_processing_metrics,
                "after_processing": final_metrics
            },
            "speakers": speaker_audio_segments
        }
        await save_request_log(performance_data)

    # ✅ Return structured response with file URLs
    return {
        "request_id": request_id,
        "file": file.filename,
        "processing_time_seconds": total_time,
        "speakers": speaker_audio_segments
    }

@app.get("/logs")
async def list_logs():
    """
    Lists all saved performance logs.
    """
    try:
        logs = [f for f in os.listdir("logs") if f.endswith(".json")]
        return {"logs": logs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to list logs: {str(e)}"})

@app.get("/logs/{log_filename}")
async def get_log(log_filename: str):
    """
    Returns a specific performance log by filename.
    """
    log_path = os.path.join("logs", log_filename)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as log_file:
            return json.load(log_file)
    return JSONResponse(status_code=404, content={"error": "Log file not found"})

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
