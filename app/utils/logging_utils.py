import os
import json
import time
import uuid
import psutil
import datetime

# ✅ Ensure logs directory exists
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def get_system_metrics():
    """
    Captures CPU, RAM, Disk, Thread, and Process usage.

    Returns:
    - dict: System resource metrics.
    """
    return {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "ram_usage_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage("/").percent,
        "active_threads": len(psutil.Process().threads()),
        "active_processes": len(psutil.pids())
    }

def save_request_log(data: dict):
    """
    Saves the given request performance data as a JSON log file.

    Args:
    - data (dict): The request details and performance metrics.

    Returns:
    - str: Path to the saved log file.
    """
    request_id = data.get("request_id", str(uuid.uuid4()))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_filename = f"log_{timestamp}_{request_id}.json"
    log_path = os.path.join(LOGS_DIR, log_filename)

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            json.dump(data, log_file, indent=4)
        return log_path
    except Exception as e:
        print(f"🚨 Failed to save log file: {e}")
        return None
