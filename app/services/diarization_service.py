import logging
import io
import time
import torch
import os
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline
from app.utils.audio_utils import (
    convert_audio_format,
    merge_speaker_segments,
    extract_speaker_segments
)
from app.utils.config import get_huggingface_token

# ✅ Load Environment Variable
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"  # Convert to boolean

class DiarizationProcessor:
    def __init__(self):
        self.pipeline = self.load_pipeline()
        torch.set_grad_enabled(False)  # ✅ Disable gradient computation globally for inference speedup

    def load_pipeline(self):
        """Load Pyannote model and respect GPU/CPU setting."""
        logging.info("⏳ Loading Pyannote Speaker Diarization Model...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=get_huggingface_token())

        if USE_GPU and torch.cuda.is_available():
            device = torch.device("cuda")
            pipeline.to(device)  # ✅ Move model to GPU
            logging.info("🚀 Pyannote model moved to GPU.")
        else:
            device = torch.device("cpu")
            pipeline.to(device)  # ✅ Force CPU mode
            logging.info("⚡ Pyannote model running on CPU.")

        self.device = device  # ✅ Store device reference for input processing
        return pipeline

    async def process_audio(self, file, request_id):
        """Process audio file for speaker diarization."""
        step_timings = {}

        # ✅ Track File Metadata
        file_metadata = {
            "file_name": file.filename,
            "file_size_bytes": file.size
        }

        # ✅ Read and Load Audio Once
        original_audio = await file.read()
        file_metadata["file_size_bytes"] = len(original_audio)

        # ✅ Step 1: Load Audio into NumPy Array
        step_1_start = time.time()
        audio_data, sample_rate = sf.read(io.BytesIO(original_audio), dtype="int16")
        step_timings["audio_loading"] = time.time() - step_1_start

        # ✅ Step 2: Validate and Convert if Needed
        step_2_start = time.time()
        if sample_rate != 16000 or audio_data.ndim > 1:
            audio_data = convert_audio_format(audio_data, sample_rate)
        step_timings["audio_conversion"] = time.time() - step_2_start

        # ✅ Move Input Audio to GPU if enabled
        step_3_start = time.time()
        if USE_GPU:
            audio_tensor = torch.tensor(audio_data).to(self.device)
        step_timings["audio_gpu_transfer"] = time.time() - step_3_start

        # ✅ Step 4: Speaker Diarization Processing
        step_4_start = time.time()
        with torch.no_grad():
            diarization_result = self.pipeline(io.BytesIO(original_audio))
        step_timings["diarization_processing"] = time.time() - step_4_start

        # ✅ Step 5: Extract Speakers & Merge Segments
        step_5_start = time.time()
        raw_segments = [
            {"speaker": speaker, "start": round(segment.start, 2), "end": round(segment.end, 2)}
            for segment, _, speaker in diarization_result.itertracks(yield_label=True)
        ]
        merged_segments = merge_speaker_segments(raw_segments)
        step_timings["segmentation_extraction"] = time.time() - step_5_start

        # ✅ Step 6: Extract & Save Speaker-Specific Audio Clips
        step_6_start = time.time()
        speaker_audio_segments = extract_speaker_segments(original_audio, merged_segments, "saved_audio", "http://localhost:7000/audio")
        step_timings["segment_extraction"] = time.time() - step_6_start

        # ✅ Step 7: Capture GPU Metrics
        gpu_metrics = {"used": USE_GPU}
        if USE_GPU and torch.cuda.is_available():
            gpu_metrics.update({
                "gpu_usage_percent": torch.cuda.utilization(0),
                "gpu_memory_used_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
                "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2)
            })

        return {
            "request_id": request_id,
            "file_metadata": file_metadata,
            "speakers": speaker_audio_segments,
            "step_timings": step_timings,
            "gpu_metrics": gpu_metrics
        }
