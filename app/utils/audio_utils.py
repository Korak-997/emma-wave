import io
import uuid
import base64
import logging
import soundfile as sf
import numpy as np
import ffmpeg
import os

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

def extract_speaker_segments(original_audio, segments, save_path, audio_url_base):
    """
    Extracts and assigns audio segments to respective speakers.

    Args:
    - original_audio (bytes): The original full audio in bytes.
    - segments (list): List of merged speaker segments [{speaker, start, end}]
    - save_path (str): Path to store extracted audio files.
    - audio_url_base (str): Base URL for accessing audio files.

    Returns:
    - Dictionary with speaker-wise segmented audio
    """
    try:
        import os
        # ✅ Load original audio into memory
        audio_buffer = io.BytesIO(original_audio)
        audio_data, samplerate = sf.read(audio_buffer, dtype="int16")

        speaker_clips = {}

        for index, segment in enumerate(segments, start=1):
            speaker = segment["speaker"]
            start_sample = int(segment["start"] * samplerate)
            end_sample = int(segment["end"] * samplerate)

            # ✅ Extract segment without modifying the original
            segment_audio = audio_data[start_sample:end_sample]

            # ✅ Define file path
            file_name = f"segment_{index}.wav"
            file_path = os.path.join(save_path, file_name)

            # ✅ Save as WAV file
            sf.write(file_path, segment_audio, samplerate, format="WAV")

            # ✅ Assign unique ID and store segment metadata
            segment_entry = {
                "id": str(uuid.uuid4()),
                "audio_url": f"{audio_url_base}/{file_name}",
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
