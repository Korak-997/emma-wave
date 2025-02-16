import os
import logging
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

def get_huggingface_token():
    """Fetch the Hugging Face token from environment variables."""
    token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
    if not token:
        logging.error("🚨 HUGGINGFACE_TOKEN is missing! Make sure it's set in your .env file.")
        raise ValueError("HUGGINGFACE_TOKEN is not set!")
    return token
