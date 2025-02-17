import os
import json
import uuid
import psutil
import datetime
import aiofiles  # Use async file writing for better performance

# ✅ Load environment variable for logging control
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"

# ✅ Ensure logs directory exists if logging is enabled
LOGS_DIR = "logs"
if ENABLE_LOGGING:
    os.makedirs(LOGS_DIR, exist_ok=True)

def get_system_metrics():
    """
    Captures CPU, RAM, Disk, Thread, and Process usage.

    Returns:
    - dict: System resource metrics.
    """
    return {
        "cpu_usage_percent": psutil.cpu_percent(interval=0.1),  # Reduce delay
        "ram_usage_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage("/").percent,
        "active_threads": len(psutil.Process().threads()),
        "active_processes": len(psutil.pids())
    }

async def save_request_log(data: dict):
    """
    Saves the given request performance data as a JSON log file asynchronously.

    Args:
    - data (dict): The request details and performance metrics.

    Returns:
    - str: Path to the saved log file or None if logging is disabled.
    """
    if not ENABLE_LOGGING:
        return None  # Skip logging if disabled

    request_id = data.get("request_id", str(uuid.uuid4()))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_filename = f"log_{timestamp}_{request_id}.json"
    log_path = os.path.join(LOGS_DIR, log_filename)

    try:
        async with aiofiles.open(log_path, "w", encoding="utf-8") as log_file:
            await log_file.write(json.dumps(data, indent=4))
            print(f"__Logs generated and saved successfully__")
        return log_path
    except Exception as e:
        print(f"🚨 Failed to save log file: {e}")
        return None
