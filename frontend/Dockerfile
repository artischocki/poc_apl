FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies (cached layer — only re-runs when pyproject.toml changes)
COPY pyproject.toml .
RUN uv sync --no-cache

# Copy source
COPY . .

# Directory for the SQLite persistence volume
RUN mkdir -p /app/data

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 3000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "3000"]
