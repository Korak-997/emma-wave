import os
import json
import datetime
import aiofiles

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

class LoggingService:
    async def save_log(self, request_id, data):
        """Save log as a JSON file asynchronously."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_filename = f"log_{timestamp}_{request_id}.json"
        log_path = os.path.join(LOGS_DIR, log_filename)

        async with aiofiles.open(log_path, "w", encoding="utf-8") as log_file:
            await log_file.write(json.dumps(data, indent=4))
