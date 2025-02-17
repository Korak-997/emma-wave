import os
import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse

LOGS_DIR = "logs"
router = APIRouter(prefix="/logs", tags=["Logs"])

@router.get("/")
async def list_logs():
    """List all log files."""
    try:
        logs = [f for f in os.listdir(LOGS_DIR) if f.endswith(".json")]
        return {"logs": logs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to list logs: {str(e)}"})

@router.get("/{log_filename}")
async def get_log(log_filename: str):
    """Retrieve a specific log file."""
    log_path = os.path.join(LOGS_DIR, log_filename)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as log_file:
            return json.load(log_file)
    return JSONResponse(status_code=404, content={"error": "Log file not found"})
