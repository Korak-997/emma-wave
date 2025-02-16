import logging
import subprocess
import soundfile as sf
import io
from app.utils.exceptions import InvalidAudioFormatError, AudioProcessingError

# ✅ Validate audio format (In-Memory)
def validate_audio_format(audio_bytes):
    """
    Checks if the audio is already in the correct format: 16-bit PCM, 16kHz, mono.
    Returns True if valid, False otherwise.
    """
    try:
        audio_file = io.BytesIO(audio_bytes)  # Read audio from memory
        data, samplerate = sf.read(audio_file)
        if samplerate == 16000 and data.ndim == 1:
            logging.info("✅ Audio is already in the correct format.")
            return True
        else:
            logging.warning(f"⚠️ Audio needs conversion (Found: {samplerate}Hz, Channels: {data.ndim})")
            return False
    except Exception as e:
        logging.error(f"🚨 Could not read audio file: {e}")
        raise InvalidAudioFormatError("Unable to read the audio file. Ensure it's a valid audio format.")

# ✅ Convert audio to required format (In-Memory)
def convert_audio_format(audio_bytes):
    """
    Converts an in-memory audio file to 16-bit PCM, 16kHz, mono using ffmpeg.
    Returns the converted audio as bytes.
    """
    try:
        input_audio = io.BytesIO(audio_bytes)  # Read from memory
        output_audio = io.BytesIO()

        # Run ffmpeg conversion
        process = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
            input=input_audio.read(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        output_audio.write(process.stdout)
        output_audio.seek(0)  # Reset cursor

        logging.info("✅ Audio successfully converted in-memory.")
        return output_audio.getvalue()  # Return bytes
    except subprocess.CalledProcessError as e:
        logging.error(f"🚨 Audio conversion failed: {e}")
        raise AudioProcessingError("Audio format conversion failed. Make sure FFmpeg is installed and working.")
