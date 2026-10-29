# Test, assessment and MCP-server image. Audio, Meet and Ollama stay native on macOS:
# containers get no Metal GPU or Core Audio devices.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-install-project
COPY src ./src
COPY question_bank ./question_bank
COPY examples ./examples
COPY tests ./tests
RUN uv sync --frozen

# Reach Ollama on the host (Docker Desktop): docker run -e OLLAMA_HOST=http://host.docker.internal:11434
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENTRYPOINT ["uv", "run"]
CMD ["pytest", "-q"]
