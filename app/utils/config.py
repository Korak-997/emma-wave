import os
import logging
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

from app.utils.exceptions import ModelLoadingError

def get_huggingface_token():
    """Fetch the Hugging Face token from environment variables."""
    token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
    if not token:
        logging.error("🚨 HUGGINGFACE_TOKEN is missing! Make sure it's set in your .env file.")
        raise ModelLoadingError("Missing Hugging Face token. API cannot start.")
    return token
