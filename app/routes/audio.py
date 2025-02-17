import os
from fastapi import APIRouter
from fastapi.responses import FileResponse

AUDIO_SAVE_PATH = "saved_audio"
router = APIRouter(prefix="/audio", tags=["Audio"])

@router.get("/{filename}")
async def get_audio(filename: str):
    """Serve audio files."""
    file_path = os.path.join(AUDIO_SAVE_PATH, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(file_path, media_type="audio/wav")
