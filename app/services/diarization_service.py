import logging
import io
import torch
from pyannote.audio import Pipeline
from app.utils.audio_utils import validate_audio_format, convert_audio_format, merge_speaker_segments, extract_speaker_segments
from app.utils.config import get_huggingface_token

USE_GPU = torch.cuda.is_available()

class DiarizationProcessor:
    def __init__(self):
        self.pipeline = self.load_pipeline()

    def load_pipeline(self):
        """Load Pyannote model and move to GPU if available."""
        logging.info("⏳ Loading Pyannote Speaker Diarization Model...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=get_huggingface_token())
        if USE_GPU:
            pipeline.to(torch.device("cuda"))
            logging.info("🚀 Pyannote model moved to GPU.")
        return pipeline

    async def process_audio(self, file, request_id):
        """Process audio file for speaker diarization."""
        original_audio = await file.read()
        if not validate_audio_format(original_audio):
            original_audio = convert_audio_format(original_audio)

        diarization_result = self.pipeline(io.BytesIO(original_audio))

        # Extract speakers
        raw_segments = [
            {"speaker": speaker, "start": round(segment.start, 2), "end": round(segment.end, 2)}
            for segment, _, speaker in diarization_result.itertracks(yield_label=True)
        ]
        merged_segments = merge_speaker_segments(raw_segments)
        speaker_audio_segments = extract_speaker_segments(original_audio, merged_segments, "saved_audio", "http://localhost:7000/audio")

        return {"request_id": request_id, "speakers": speaker_audio_segments}
