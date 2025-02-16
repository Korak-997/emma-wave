import os
import logging
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.utils.exceptions import InvalidAudioFormatError, AudioProcessingError, ModelLoadingError

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Create a test client for FastAPI
client = TestClient(app)

# ✅ Define invalid test files
INVALID_FILE = "tests/invalid_file.txt"  # Non-audio file
CORRUPT_AUDIO = "tests/corrupt_audio.wav"  # Corrupt or unreadable audio file

@pytest.fixture(scope="module", autouse=True)
def setup_test_files():
    """Fixture to create test files before running tests."""
    if not os.path.exists("tests"):
        os.makedirs("tests")

    # Create a fake text file
    with open(INVALID_FILE, "w") as f:
        f.write("This is not an audio file.")

    # Create an empty (corrupt) audio file
    with open(CORRUPT_AUDIO, "wb") as f:
        f.write(b"\x00\x00\x00\x00")

    yield

    # Cleanup after tests
    os.remove(INVALID_FILE)
    os.remove(CORRUPT_AUDIO)

def test_invalid_audio_format():
    """
    Test that the API rejects non-audio files.
    """
    logging.info("🚀 Running Invalid Audio Format Test...")

    with open(INVALID_FILE, "rb") as f:
        files = {"file": ("invalid_file.txt", f, "text/plain")}
        response = client.post("/diarize", files=files)

    assert response.status_code == 400, "❌ Expected 400 status code for invalid file."
    assert "Unable to read the audio file" in response.json()["detail"], "❌ Incorrect error message."

    logging.info("✅ Invalid Audio Format Test Passed!")


def test_corrupt_audio_file():
    """
    Test that the API handles corrupt or unreadable audio files.
    """
    logging.info("🚀 Running Corrupt Audio File Test...")

    with open(CORRUPT_AUDIO, "rb") as f:
        files = {"file": ("corrupt_audio.wav", f, "audio/wav")}
        response = client.post("/diarize", files=files)

    assert response.status_code == 400, "❌ Expected 400 status code for corrupt audio file."
    assert "Unable to read the audio file" in response.json()["detail"], "❌ Incorrect error message."

    logging.info("✅ Corrupt Audio File Test Passed!")


def test_model_loading_error(monkeypatch):
    """
    Test that the API handles missing Hugging Face token properly.
    """
    logging.info("🚀 Running Model Loading Error Test...")

    # Simulate missing Hugging Face token
    monkeypatch.delenv("HUGGING_FACE_ACCESS_TOKEN", raising=False)

    # Restart the app to simulate environment reset
    with pytest.raises(ModelLoadingError, match="Missing Hugging Face token. API cannot start."):
        from app.utils.config import get_huggingface_token
        get_huggingface_token()

    logging.info("✅ Model Loading Error Test Passed!")
