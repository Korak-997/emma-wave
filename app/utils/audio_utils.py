import io
import uuid
import base64
import logging
import soundfile as sf
import numpy as np
import ffmpeg

from app.utils.exceptions import AudioProcessingError

def convert_audio_format(input_audio):
    """
    Converts audio to 16-bit PCM, 16kHz, mono if necessary.

    Args:
    - input_audio (bytes): The input audio file in bytes.

    Returns:
    - Converted audio in bytes (16-bit PCM, 16kHz, mono)
    """
    try:
        audio_buffer = io.BytesIO(input_audio)

        # ✅ Read audio using SoundFile
        audio_data, samplerate = sf.read(audio_buffer, dtype="int16")

        # ✅ Check format (Must be: 16-bit PCM, 16kHz, mono)
        if samplerate == 16000 and audio_data.ndim == 1:
            logging.info("✅ Audio is already in the correct format.")
            return input_audio

        logging.warning(f"⚠️ Audio needs conversion (Found: {samplerate}Hz, Channels: {audio_data.shape[1] if audio_data.ndim > 1 else 1})")

        # ✅ Convert using FFmpeg
        output_buffer = io.BytesIO()
        process = (
            ffmpeg
            .input("pipe:0")
            .output("pipe:1", format="wav", acodec="pcm_s16le", ar="16000", ac="1")
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        output_audio, _ = process.communicate(input_audio)

        logging.info("✅ Audio successfully converted in-memory.")
        return output_audio

    except Exception as e:
        logging.error(f"🚨 Audio conversion failed: {e}")
        raise AudioProcessingError("Audio format conversion failed.")

# ✅ Extract speaker segments without modifying the original file
def extract_speaker_segments(original_audio, segments):
    """
    Extracts and assigns audio segments to respective speakers.

    Args:
    - original_audio (bytes): The original full audio in bytes.
    - segments (list): List of merged speaker segments [{speaker, start, end}]

    Returns:
    - Dictionary with speaker-wise segmented audio
    """
    try:
        # ✅ Load original audio into memory
        audio_buffer = io.BytesIO(original_audio)
        audio_data, samplerate = sf.read(audio_buffer, dtype="int16")

        speaker_clips = {}

        for segment in segments:
            speaker = segment["speaker"]
            start_sample = int(segment["start"] * samplerate)
            end_sample = int(segment["end"] * samplerate)

            # ✅ Extract segment without modifying the original
            segment_audio = audio_data[start_sample:end_sample]

            # ✅ Save in memory as WAV format
            output_buffer = io.BytesIO()
            sf.write(output_buffer, segment_audio, samplerate, format="WAV")
            output_buffer.seek(0)

            # ✅ Assign unique ID and store segment
            segment_entry = {
                "id": str(uuid.uuid4()),
                "audio": base64.b64encode(output_buffer.getvalue()).decode("utf-8"),  # ✅ Encode in Base64
                "start": segment["start"],
                "end": segment["end"]
            }

            if speaker not in speaker_clips:
                speaker_clips[speaker] = []

            speaker_clips[speaker].append(segment_entry)

        logging.info("✅ Successfully extracted and assigned speaker segments.")
        return speaker_clips

    except Exception as e:
        logging.error(f"🚨 Error extracting speaker segments: {e}")
        raise AudioProcessingError("Failed to extract speaker segments.")



def validate_audio_format(input_audio):
    """
    Validates if the audio is in the correct format: 16-bit PCM, 16kHz, mono.

    Args:
    - input_audio (bytes): The input audio file in bytes.

    Returns:
    - (bool): True if the format is correct, otherwise False.
    """
    try:
        audio_buffer = io.BytesIO(input_audio)
        audio_data, samplerate = sf.read(audio_buffer, dtype="int16")

        # ✅ Check format
        if samplerate == 16000 and audio_data.ndim == 1:
            return True
        return False

    except Exception as e:
        logging.error(f"🚨 Audio validation failed: {e}")
        return False



def merge_speaker_segments(segments, gap_threshold=0.5):
    """
    Merges consecutive speaker segments if the gap between them is below a threshold.

    Args:
    - segments (list): List of speaker segments [{speaker, start, end}]
    - gap_threshold (float): Maximum gap (in seconds) allowed for merging.

    Returns:
    - List of merged speaker segments
    """
    if not segments:
        return []

    # ✅ Sort segments by start time
    segments.sort(key=lambda x: x["start"])

    merged_segments = []
    current_segment = segments[0]

    for next_segment in segments[1:]:
        if current_segment["speaker"] == next_segment["speaker"] and \
           (next_segment["start"] - current_segment["end"]) <= gap_threshold:
            # ✅ Merge segments
            current_segment["end"] = next_segment["end"]
        else:
            merged_segments.append(current_segment)
            current_segment = next_segment

    merged_segments.append(current_segment)

    logging.info(f"✅ Merged {len(segments)} segments into {len(merged_segments)}.")
    return merged_segments
