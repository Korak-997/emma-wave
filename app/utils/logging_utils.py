import os
import json
import uuid
import psutil
import datetime
import aiofiles
import torch
import asyncio
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



async def get_system_metrics():
    """
    Captures CPU, RAM, Disk, Thread, Process, and GPU usage in parallel.
    """
    async def fetch_cpu_usage():
        return psutil.cpu_percent(interval=0.1)

    async def fetch_ram_usage():
        return psutil.virtual_memory().percent

    async def fetch_disk_usage():
        return psutil.disk_usage("/").percent

    async def fetch_gpu_usage():
        if not torch.cuda.is_available():
            return None  # No GPU available

        try:
            from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetHandleByIndex
            handle = nvmlDeviceGetHandleByIndex(0)
            return nvmlDeviceGetUtilizationRates(handle).gpu
        except Exception:
            return None  # If GPU tracking fails, return None

    # ✅ Run all system metric collection in parallel
    results = await asyncio.gather(
        fetch_cpu_usage(),
        fetch_ram_usage(),
        fetch_disk_usage(),
        fetch_gpu_usage()
    )

    return {
        "cpu_usage_percent": results[0],
        "ram_usage_percent": results[1],
        "disk_usage_percent": results[2],
        "gpu_usage_percent": results[3],
    }


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
