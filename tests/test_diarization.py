import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create a test client for FastAPI
client = TestClient(app)

# Define the path to the test audio file
TEST_AUDIO_FILE = "tests/sample_audio.wav"

# Debugging: Check if the file exists
print(f"📂 Checking if test audio file exists: {TEST_AUDIO_FILE}")
if not os.path.exists(TEST_AUDIO_FILE):
    print("🚨 ERROR: Test audio file not found! Make sure `tests/sample_audio.wav` exists.")

# Skip the test if the file is missing
@pytest.mark.skipif(not os.path.exists(TEST_AUDIO_FILE), reason="Test audio file not found.")
def test_diarization_api():
    """
    Test the diarization API by sending an audio file and checking the response format.
    """

    print("🚀 Running Diarization API Test...")

    # Open the test audio file and send it to the API
    with open(TEST_AUDIO_FILE, "rb") as audio:
        files = {"file": ("sample_audio.wav", audio, "audio/wav")}
        response = client.post("/diarize", files=files)

    print("📡 Sent request to /diarize endpoint.")

    # Debugging: Print response
    print("🔄 Response Status Code:", response.status_code)
    print("📜 Response JSON:", response.json())

    # Check if the request was successful
    assert response.status_code == 200, f"❌ API call failed! Status code: {response.status_code}"

    # Check if the response contains the "segments" key
    response_json = response.json()
    assert "segments" in response_json, "❌ Response does not contain 'segments' key."

    # Ensure at least one speaker is detected
    assert len(response_json["segments"]) > 0, "❌ No speakers were detected!"

    # Validate the structure of each segment
    for i, segment in enumerate(response_json["segments"]):
        assert "speaker" in segment, f"❌ Missing 'speaker' key in segment {i}"
        assert "start" in segment, f"❌ Missing 'start' key in segment {i}"
        assert "end" in segment, f"❌ Missing 'end' key in segment {i}"

    print("✅ Test Passed: Diarization API is working correctly!")
