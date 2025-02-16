import os
import logging
import subprocess
import soundfile as sf
from app.utils.exceptions import InvalidAudioFormatError, AudioProcessingError

# ✅ Validate audio format
def validate_audio_format(input_file):
    """
    Checks if the audio is already in the correct format: 16-bit PCM, 16kHz, mono.
    Returns True if valid, False otherwise.
    """
    try:
        data, samplerate = sf.read(input_file)
        if samplerate == 16000 and data.ndim == 1:
            logging.info(f"✅ Audio {input_file} is already in the correct format.")
            return True
        else:
            logging.warning(f"⚠️ Audio {input_file} needs conversion (Found: {samplerate}Hz, Channels: {data.ndim})")
            return False
    except Exception as e:
        logging.error(f"🚨 Could not read audio file {input_file}: {e}")
        raise InvalidAudioFormatError("Unable to read the audio file. Ensure it's a valid audio format.")

# ✅ Convert audio to required format
def convert_audio_format(input_file):
    """
    Converts the audio file to 16-bit PCM, 16kHz, mono using ffmpeg.
    Returns the new filename if successful, raises error otherwise.
    """
    output_file = f"converted_{os.path.basename(input_file)}"
    command = [
        "ffmpeg", "-i", input_file,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_file
    ]
    try:
        subprocess.run(command, check=True)
        logging.info(f"✅ Audio successfully converted: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logging.error(f"🚨 Audio conversion failed: {e}")
        raise AudioProcessingError("Audio format conversion failed. Make sure FFmpeg is installed and working.")
