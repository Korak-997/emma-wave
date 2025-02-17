import os
import json
import uuid
import psutil
import datetime
import aiofiles
import torch

# ✅ Load environment variable for logging control
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"

# ✅ Ensure logs directory exists if logging is enabled
LOGS_DIR = "logs"
if ENABLE_LOGGING:
    os.makedirs(LOGS_DIR, exist_ok=True)

# ✅ Try Importing pynvml for GPU Usage Tracking
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
    nvmlInit()
    NVML_AVAILABLE = True
except ModuleNotFoundError:
    NVML_AVAILABLE = False
    print("⚠️ `pynvml` is not installed. GPU usage metrics will not be available.")
except Exception as e:
    NVML_AVAILABLE = False
    print(f"⚠️ Failed to initialize NVML: {e}")

def get_system_metrics():
    """
    Captures CPU, RAM, Disk, and GPU usage metrics.
    """
    metrics = {
        "cpu_usage_percent": psutil.cpu_percent(interval=0.1),  # Reduce delay
        "ram_usage_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage("/").percent,
        "active_threads": len(psutil.Process().threads()),
        "active_processes": len(psutil.pids())
    }

    # ✅ Capture GPU Usage If Available
    if torch.cuda.is_available() and NVML_AVAILABLE:
        try:
            handle = nvmlDeviceGetHandleByIndex(0)
            gpu_utilization = nvmlDeviceGetUtilizationRates(handle)
            gpu_memory = nvmlDeviceGetMemoryInfo(handle)

            metrics["gpu_metrics"] = {
                "gpu_usage_percent": gpu_utilization.gpu,
                "gpu_memory_used_mb": round(gpu_memory.used / (1024 * 1024), 2),
                "gpu_memory_total_mb": round(gpu_memory.total / (1024 * 1024), 2)
            }
        except Exception as e:
            print(f"⚠️ Failed to retrieve GPU metrics: {e}")

    return metrics

async def save_request_log(data: dict):
    """
    Saves the given request performance data as a JSON log file asynchronously.
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
