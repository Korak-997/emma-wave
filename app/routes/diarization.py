import logging
import uuid
import time
import datetime
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.services.diarization_service import DiarizationProcessor
from app.services.logging_service import LoggingService
from app.utils.exceptions import InvalidAudioFormatError, AudioProcessingError
from app.utils.logging_utils import get_system_metrics

router = APIRouter(prefix="/diarize", tags=["Diarization"])
diarization_processor = DiarizationProcessor()
logging_service = LoggingService()

@router.post("/")
async def diarize_audio(file: UploadFile = File(...)):
    """Process uploaded audio file for speaker diarization."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # ✅ Capture initial system metrics
    initial_metrics = get_system_metrics()

    logging.info(f"📥 Received file: {file.filename}, Content-Type: {file.content_type}")

    try:
        # ✅ Process audio
        result = await diarization_processor.process_audio(file, request_id)
        result["processing_time_seconds"] = time.time() - start_time

        # ✅ Capture final system metrics
        result["system_metrics"] = {
            "before_processing": initial_metrics,
            "after_processing": get_system_metrics(),
        }

        # ✅ Save logs
        await logging_service.save_log(request_id, result)

        return result

    except (InvalidAudioFormatError, AudioProcessingError) as e:
        logging.error(f"🚨 Processing Error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

    except Exception as e:
        logging.error(f"🚨 Unexpected Error: {e}")
        return JSONResponse(status_code=500, content={"error": "Unexpected error occurred."})
