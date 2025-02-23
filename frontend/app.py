"""Streamlit frontend for the SpeechBot application."""

import time
from typing import Any, Dict, List

import requests
import streamlit as st

SPEAKERS = ["Jon", "Lea", "Gary", "Jenna", "Mike"]


def check_service_health(service_url: str) -> bool:
    """Check if a FastAPI service is running by fetching its OpenAPI spec.

    Parameters
    ----------
    service_url : str
        The base URL of the service.

    Returns
    -------
    bool
        True if the service is up, False otherwise.
    """
    try:
        response = requests.get(f"{service_url}/openapi.json", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def wait_for_services(stt_url: str, llm_url: str, tts_url: str) -> None:
    """Display a waiting UI until all required services are up.

    Parameters
    ----------
    stt_url : str
        URL for the Speech-To-Text service.
    llm_url : str
        URL for the Language Model service.
    tts_url : str
        URL for the Text-To-Speech service.
    """
    placeholder = st.empty()
    stt_is_up = check_service_health(stt_url)
    llm_is_up = check_service_health(llm_url)
    tts_is_up = check_service_health(tts_url)
    while not all((stt_is_up, llm_is_up, tts_is_up)):
        with placeholder.container():
            st.markdown("## Waiting for Services...")
            st.write(f"- **STT** ({stt_url}): {'✅' if stt_is_up else '⏳'}")
            st.write(f"- **LLM** ({llm_url}): {'✅' if llm_is_up else '⏳'}")
            st.write(f"- **TTS** ({tts_url}): {'✅' if tts_is_up else '⏳'}")

        time.sleep(2)
        stt_is_up = check_service_health(stt_url)
        llm_is_up = check_service_health(llm_url)
        tts_is_up = check_service_health(tts_url)

    st.rerun()


def generate_speech(prompt: str, speaker_description: str, tts_url: str) -> bytes:
    """Call the FastAPI service to generate speech audio based on the prompt and speaker description.

    Parameters
    ----------
    prompt : str
        The text prompt for generating speech.
    speaker_description : str
        A description of the speaker for the TTS model.
    tts_url : str
        The URL of the TTS service.

    Returns
    -------
    bytes
        The WAV audio file in bytes.

    Raises
    ------
    Exception
        Exception: If the API call fails.

    Example
    -------

    >>> audio_bytes = generate_speech("Hello there!", "A friendly female voice.")
    """
    payload = {"prompt": prompt, "speaker_description": speaker_description}
    response = requests.post(f"{tts_url}/generate_speech/", json=payload)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")


def send_chat_request(messages: List[Dict[str, str]], llm_url: str) -> str:
    """Send a chat request to the FastAPI service and return the assistant's response.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        The list of chat messages.
    llm_url : str
        The URL of the LLM service.

    Returns
    -------
    str
        The assistant's chat response.
    """
    payload: Dict[str, Any] = {"messages": messages}
    response = requests.post(f"{llm_url}/chat", json=payload)
    if response.status_code == 200:
        data: Dict[str, Any] = response.json()
        return data.get("answer", "")
    else:
        st.error("Error from API: " + response.text)
        return ""


def transcribe_speech(audio_file: bytes, stt_url: str) -> str:
    """Transcribe an audio file using the FastAPI service.

    Parameters
    ----------
    audio_file : bytes
        The audio file in bytes.
    stt_url : str
        The URL of the Speech To Text service.

    Returns
    -------
    str
        The transcription of the audio file.

    Raises
    ------
    Exception
        If the API call fails.

    Example
    -------

    >>> transcription = transcribe_speech(audio_file)
    """
    files = {"file": ("recording.wav", audio_file, "audio/wav")}
    response = requests.post(f"{stt_url}/transcribe", files=files)
    if response.status_code == 200:
        return response.json().get("transcription", "")
    else:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="SpeechBot",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Sidebar configuration
    st.sidebar.markdown("# Service URLs")
    stt_url = st.sidebar.text_input("STT URL", value="http://stt_service:8000")
    llm_url = st.sidebar.text_input("LLM URL", value="http://llm_service:8000")
    tts_url = st.sidebar.text_input("TTS URL", value="http://tts_service:8000")
    st.sidebar.markdown("# Assistant Configuration")
    assistant_role = st.sidebar.text_input("Assistant role", value="You are a helpful assistant.")
    speaker_choice = st.sidebar.selectbox("Select the assistant voice:", SPEAKERS)

    if "txt_messages" not in st.session_state:
        # Initialize conversation with a system prompt
        st.session_state["txt_messages"] = [{"role": "system", "content": assistant_role}]
        st.session_state["audio_messages"] = [None]

    if "audio_input_key_counter" not in st.session_state:
        st.session_state.audio_input_key_counter = 0

    st.title("Speak with SpeechBot")

    if not all(
        (
            check_service_health(stt_url),
            check_service_health(llm_url),
            check_service_health(tts_url),
        )
    ):
        wait_for_services(stt_url, llm_url, tts_url)

    # Display previous conversation messages
    for i, (txt, audio) in enumerate(
        zip(st.session_state["txt_messages"], st.session_state["audio_messages"])
    ):
        last = i == len(st.session_state["txt_messages"]) - 1  # Autoplay the last audio message

        if txt["role"] == "system":
            continue

        with st.chat_message(txt["role"]):
            st.audio(audio, format="audio/wav", autoplay=last)
            with st.expander("Show text"):
                st.write(txt["content"])

    audio_input_key = f"audio_input_key_{st.session_state.audio_input_key_counter}"
    input_box = st.chat_message("user")

    audio_placeholder = input_box.empty()
    user_audio_file = audio_placeholder.audio_input(
        label=" ", label_visibility="collapsed", key=audio_input_key
    )

    if user_audio_file:
        audio_placeholder.audio(user_audio_file, format="audio/wav")

        st.session_state["audio_messages"].append(user_audio_file)

        with st.chat_message("assistant"):
            with st.status("Preparing response..."):
                st.write("Transcribing audio...")

                user_txt = transcribe_speech(user_audio_file, stt_url)
                with input_box.expander("Show text"):
                    st.write(user_txt)

                # Append user message to the conversation
                st.session_state["txt_messages"].append({"role": "user", "content": user_txt})

                st.write("Computing txt response from assistant...")
                # Get the response from the FastAPI service
                answer: str = send_chat_request(st.session_state["txt_messages"], llm_url)

                st.write("Computing output audio...")

                audio_data = generate_speech(
                    answer, f"{speaker_choice} voice speaking dynamically.", tts_url
                )
                st.write("Speech generated successfully!")

                st.session_state["txt_messages"].append({"role": "assistant", "content": answer})
                st.session_state["audio_messages"].append(audio_data)

        # Hack to change when https://github.com/streamlit/streamlit/issues/9710 is resolved
        # Update so that `user_audio_file is None` the next time around
        del st.session_state[audio_input_key]
        st.session_state.audio_input_key_counter += 1

        # Rerun to update the chat history and get new input
        st.rerun()
