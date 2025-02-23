"""FastAPI service for generating speech audio using the ParlerTTS mini model."""

import io

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

app = FastAPI(
    title="ParlerTTS Service",
    description="A FastAPI service for generating speech audio using the ParlerTTS mini model.",
    version="0.1.0",
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizers
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1.1").to(
    device, dtype=torch.bfloat16
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1.1")
speaker_description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)


class TTSRequest(BaseModel):
    """Request model for TTS generation.

    Attributes
    ----------
    prompt : str
        The text prompt for generating speech.
    speaker_description : str
        A description of the speaker to condition the TTS output.
    """

    prompt: str = Field(
        description="The text prompt for generating speech.",
        example="Hello, welcome to our service!",
    )
    speaker_description: str = Field(
        description="A description of the speaker to condition the TTS output.",
        example="A friendly male voice with an American accent.",
    )


@app.post("/generate_speech/", response_class=StreamingResponse)
def generate_speech(request: TTSRequest) -> FileResponse:
    """Generate a speech audio from a text prompt and speaker description.

    This endpoint uses the ParlerTTS model to generate an audio waveform from the provided text prompt,
    conditioned on the speaker description. The resulting audio is returned as a WAV file in a streaming response.

    Parameters
    ----------
    request : TTSRequest
        The request body containing the prompt and speaker description.

    Returns
    -------
    FileResponse
        StreamingResponse: A streaming response containing the generated audio in WAV format.

    Raises
    ------
    HTTPException
        If an error occurs during generation.
    """

    try:
        with torch.no_grad():
            input_ids = speaker_description_tokenizer(
                request.speaker_description, return_tensors="pt"
            ).input_ids.to(device)
            prompt_input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids.to(device)

            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

        audio_arr = generation.to("cpu", torch.float32).numpy().squeeze()

        # Create an in-memory bytes buffer and write the audio as a WAV file
        buffer = io.BytesIO()
        sf.write(buffer, audio_arr, model.config.sampling_rate, format="WAV")
        buffer.seek(0)

        # Return the audio file as a streaming response with appropriate headers
        headers = {"Content-Disposition": "attachment; filename=speech.wav"}
        return StreamingResponse(buffer, media_type="audio/wav", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home() -> dict[str, str]:
    """Home endpoint that returns a welcome message.

    Returns
    -------
    dict[str, str]
        A dictionary with a welcome message.
    """
    return {"message": "Welcome to the ParlerTTS FastAPI service!"}
