import logging
from fastapi import APIRouter
from app.services.diarization_service import DiarizationProcessor

router = APIRouter(prefix="/health", tags=["Health"])
diarization_processor = DiarizationProcessor()

@router.get("/")
async def health_check():
    """Check system health."""
    try:
        if not diarization_processor.pipeline:
            raise RuntimeError("Diarization model is not loaded.")
        return {"status": "ok", "model": "loaded"}
    except Exception as e:
        logging.error(f"🚨 Health check failed: {e}")
        return {"status": "error", "message": str(e)}
