FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
      git \
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

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py"]
