FROM ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c AS uvbin

FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS builder

COPY --from=uvbin /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
COPY src ./src

RUN uv sync \
    --python /usr/local/bin/python \
    --locked \
    --no-dev \
    --no-editable

FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY alembic.ini /app/alembic.ini

ENV PATH="/app/.venv/bin:${PATH}" \
    HOME=/tmp \
    TMPDIR=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER 10001:10001

CMD ["python", "-m", "app.main"]
