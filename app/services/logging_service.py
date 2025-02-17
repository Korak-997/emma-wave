import os
import json
import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

class LoggingService:
    async def save_log(self, request_id, data):
        """Save log as a JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_filename = f"log_{timestamp}_{request_id}.json"
        log_path = os.path.join(LOGS_DIR, log_filename)

        with open(log_path, "w", encoding="utf-8") as log_file:
            json.dump(data, log_file, indent=4)
