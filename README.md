# SpeechBot

A set of services to create a prototype to speak with a chatbot that speaks back the answer to the user.

This repository is designed to create an application for a personal computer (with one GPU having 6Go of VRAM minium), further modifications are required to serve it for multiple users on the cloud.
The tutorial associated with this release is available on my blog here: <https://website.vincent-roger.fr/blog/2025/03-17-speechbot-starting-point/>.
More tutorials will come over my [blog](https://website.vincent-roger.fr/blog/) with next releases.

The announcement for the first version of this project is now live. Watch the video on YouTube: [Watch Now](https://youtu.be/5Cik2asxGfM).

## Services description

### Frontend

Streamlit service that interact with the user (to record speech and play the answer). It uses all other services in the same order as described here.

### Speech To Text service (stt_service)

This service is responsible for converting spoken language into text. It uses the Whisper tiny model.

### Large Language Model service (llm_service)

A chatBot (based on Qwen 2.5.1) to create text answer to the request of the user.

### Text-to-Speech (TTS) Service

This service converts the generated text response back into spoken language. It uses mini parler-tts model.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Docker GPU support - Follow the [NVIDIA Docker guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for installation.

Tested on linux using Fedora 41. Feedback on other platforms are welcomed.

## Setup

To set up the project, clone the repository and navigate to the project directory:

```zsh
git clone https://github.com/yourusername/voice-chatbot.git
cd voice-chatbot
```

## Usage

1. Build and launch the Docker containers using the command:

    ```zsh
    docker compose up --build
    ```

2. Open your web browser and go to <http://localhost:8501/>.
3. The web interface will wait for other services to be up and running before speaking with the bot.
4. Use the interface to record your speech and get a spoken response from the chatbot.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Licenses of the Models used

- Whisper tiny model: [MIT License](https://github.com/openai/whisper/blob/main/LICENSE)
- Qwen 2.5.1: [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE)
- Mini parler-tts model: [Apache License 2.0](https://github.com/huggingface/parler-tts/blob/main/LICENSE)
