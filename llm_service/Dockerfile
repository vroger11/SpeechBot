FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
      git \
      cmake \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# install uv, using some ideas from https://hynek.me/articles/docker-uv/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=auto \
    UV_PYTHON=python3.12 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# set workdir and install dependencies
WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN uv sync --locked --no-dev --no-install-project

# Copy the rest of your application code.
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
