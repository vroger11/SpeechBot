"""Speech Recognition Microservice using FastAPI and OpenAI's Whisper

This service provides a /transcribe endpoint that accepts an audio file upload.
The file is expected to be in WAV format and is processed as a buffer (BytesIO).
The audio is resampled to 16000 Hz (if needed) and transcribed using the Whisper model.
API documentation is automatically available at /docs and /redoc.
"""

import io

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from scipy.signal import resample
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Create the FastAPI app instance with title, description, and version.
app = FastAPI(
    title="Speech Recognition API",
    description="A microservice for speech recognition using OpenAI's Whisper model. "
    "Upload a WAV audio file as a binary buffer (BytesIO) and receive the transcription.",
    version="1.0.0",
)


class TranscriptionResponse(BaseModel):
    """Response model for the transcription endpoint.

    Attributes
    ----------
        transcription (str): The transcribed text generated from the audio.
    """

    transcription: str = Field(..., description="The transcribed text from the audio input.")


# Global configuration for the model.
model_name = "openai/whisper-tiny.en"
target_sr = 16000

# Load the processor and model from the pretrained checkpoint.
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = None


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe uploaded audio",
    description="Transcribes speech from an uploaded WAV audio file provided as a binary buffer (BytesIO). "
    "If the file's sampling rate differs from 16000 Hz, the audio is resampled accordingly.",
)
async def transcribe_audio(
    file: UploadFile = File(
        description="A WAV audio file to be transcribed, provided as a binary buffer."
    ),
) -> TranscriptionResponse:
    """Transcribe speech from an uploaded audio file provided as a binary buffer.

    This endpoint accepts a file upload (expected in WAV format), reads it into a BytesIO buffer,
    extracts the audio signal and sampling rate using soundfile, and uses the Whisper model to
    generate a transcription. If the audio's sampling rate differs from 16000 Hz, it is resampled.

    Parameters
    ----------
    file : UploadFile, optional
        The uploaded WAV audio file provided as a binary buffer.

    Returns
    -------
    TranscriptionResponse
        A JSON object containing the transcription of the audio.

    Raises
    ------
    HTTPException
        If an error occurs during the audio processing or transcription.
    """
    try:
        # Read file contents into a buffer.
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        # Load the audio file using soundfile.
        signal, sr = sf.read(audio_buffer, dtype="float32", always_2d=True)
    except Exception:
        raise HTTPException(
            status_code=400, detail="Failed to read the audio file. Ensure it is a valid WAV file."
        )

    # Use only the first channel if the audio is multi-channel.
    signal = signal[:, 0]

    # Resample the audio if the sampling rate is different from the target.
    if sr != target_sr:
        num_samples = int(len(signal) * target_sr / sr)
        resampled_audio = resample(signal, num_samples)
    else:
        resampled_audio = signal

    # Preprocess the audio for the Whisper model.
    preprocessed = processor(
        resampled_audio,
        sampling_rate=target_sr,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Generate token IDs from the model.
    predicted_ids = model.generate(
        preprocessed.input_features, attention_mask=preprocessed.attention_mask
    )

    # Decode the token IDs into text.
    transcription_list = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcription = transcription_list[0] if transcription_list else ""

    return TranscriptionResponse(transcription=transcription)
