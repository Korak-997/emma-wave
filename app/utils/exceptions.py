from fastapi import HTTPException

class InvalidAudioFormatError(HTTPException):
    def __init__(self, detail="Invalid audio format. Expected: 16-bit PCM, 16kHz, mono."):
        super().__init__(status_code=400, detail=detail)

class AudioProcessingError(HTTPException):
    def __init__(self, detail="Error processing the audio file."):
        super().__init__(status_code=500, detail=detail)

class ModelLoadingError(HTTPException):
    def __init__(self, detail="Failed to load speaker diarization model."):
        super().__init__(status_code=500, detail=detail)
