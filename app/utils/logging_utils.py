import os
import json
import time
import uuid
import psutil
from datetime import datetime

try:
    import torch
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, \
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetTemperature, nvmlSystemGetDriverVersion, nvmlShutdown, \
        NVML_TEMPERATURE_GPU

    # Initialize NVML (NVIDIA Management Library)
    nvmlInit()
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_HANDLE = nvmlDeviceGetHandleByIndex(0) if GPU_AVAILABLE else None

except ImportError:
    GPU_AVAILABLE = False
    GPU_HANDLE = None


# ✅ Ensure logs directory exists
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


def get_system_metrics():
    """
    Captures CPU, RAM, and Disk usage at the time of the function call.

    Returns:
    - dict: Contains CPU, RAM, and disk usage metrics.
    """
    return {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "ram_usage_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage("/").percent
    }


def get_gpu_metrics():
    """
    Captures GPU utilization, memory usage, and temperature if a GPU is available.

    Returns:
    - dict: GPU metrics (utilization, memory usage, temperature), or None if GPU is unavailable.
    """
    if not GPU_AVAILABLE:
        return {"gpu_used": False}

    return {
        "gpu_used": True,
        "gpu_utilization_percent": nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu,
        "gpu_memory_used_mb": nvmlDeviceGetMemoryInfo(GPU_HANDLE).used // (1024 * 1024),
        "gpu_temperature_celsius": nvmlDeviceGetTemperature(GPU_HANDLE, NVML_TEMPERATURE_GPU)
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
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_filename = f"log_{timestamp}_{request_id}.json"
    log_path = os.path.join(LOGS_DIR, log_filename)

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            json.dump(data, log_file, indent=4)
        return log_path
    except Exception as e:
        print(f"🚨 Failed to save log file: {e}")
        return None
