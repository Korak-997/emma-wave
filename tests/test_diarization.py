import os
import logging
import pytest
from fastapi.testclient import TestClient
from app.main import app

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Create a test client for FastAPI
client = TestClient(app)

# ✅ Define the path to the test audio file
TEST_AUDIO_FILE = "tests/sample_audio.wav"

# ✅ Check if the test audio file exists
if not os.path.exists(TEST_AUDIO_FILE):
    logging.error("🚨 ERROR: Test audio file not found! Make sure `tests/sample_audio.wav` exists.")

@pytest.mark.skipif(not os.path.exists(TEST_AUDIO_FILE), reason="🚨 Skipping: Test audio file not found.")
def test_diarization_api():
    """
    Test the diarization API by sending an audio file and checking the response format.
    """

    logging.info("🚀 Running Diarization API Test...")

    # ✅ Open the test audio file and send it to the API
    with open(TEST_AUDIO_FILE, "rb") as audio:
        files = {"file": ("sample_audio.wav", audio, "audio/wav")}
        response = client.post("/diarize", files=files)

    logging.info("📡 Sent request to /diarize endpoint.")

    # ✅ Ensure response is valid JSON
    assert response.headers["content-type"] == "application/json", "❌ Response is not JSON!"

    # ✅ Check if the request was successful
    assert response.status_code == 200, f"❌ API call failed! Status code: {response.status_code}"

    # ✅ Parse the JSON response
    response_json = response.json()
    logging.info(f"📜 Response JSON: {response_json}")

    # ✅ Validate response structure
    assert "segments" in response_json, "❌ Response does not contain 'segments' key."
    assert isinstance(response_json["segments"], list), "❌ 'segments' key must be a list."

    # ✅ Ensure at least one speaker is detected
    assert len(response_json["segments"]) > 0, "❌ No speakers were detected!"

    # ✅ Validate each segment's structure
    for i, segment in enumerate(response_json["segments"]):
        assert "speaker" in segment, f"❌ Missing 'speaker' key in segment {i}"
        assert "start" in segment, f"❌ Missing 'start' key in segment {i}"
        assert "end" in segment, f"❌ Missing 'end' key in segment {i}"
        assert isinstance(segment["start"], (int, float)), f"❌ 'start' must be a number in segment {i}"
        assert isinstance(segment["end"], (int, float)), f"❌ 'end' must be a number in segment {i}"

    logging.info("✅ Test Passed: Diarization API is working correctly!")


def test_cors_allowed_origin():
    """
    Test if the API correctly handles allowed CORS requests.
    """
    logging.info("🚀 Testing CORS with allowed origin...")

    headers = {"Origin": "http://localhost:3000"}  # Simulating frontend request
    response = client.options("/diarize", headers=headers)

    assert "access-control-allow-origin" in response.headers, "❌ No CORS headers in response!"
    assert response.headers["access-control-allow-origin"] == "*", "❌ CORS policy not allowing requests!"

    logging.info("✅ CORS Allowed Origin Test Passed!")


def test_cors_blocked_origin():
    """
    Test if the API rejects CORS requests from unauthorized origins.
    """
    logging.info("🚀 Testing CORS with blocked origin...")

    headers = {"Origin": "http://unauthorized-site.com"}  # Simulating unauthorized request
    response = client.options("/diarize", headers=headers)

    assert "access-control-allow-origin" in response.headers, "❌ No CORS headers in response!"
    assert response.headers["access-control-allow-origin"] == "*", "❌ CORS should reject unauthorized origin!"

    logging.info("✅ CORS Blocked Origin Test Passed!")
