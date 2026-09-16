# syntax=docker/dockerfile:1.7

# ---------- Stage 1: builder ----------
FROM python:3.14-slim-bookworm AS builder

# uv vem da imagem oficial com tag fixada (sem download via curl/wget).
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Camada de dependências: só invalida quando o lock/manifesto muda.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- Stage 2: runtime ----------
FROM python:3.14-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src

# Usuário não-root dedicado.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --home-dir /app --no-create-home app

COPY --from=builder --chown=app:app /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app src/ ./src/
# PDF default embutido; sobrescreva em runtime com
#   -v "$(pwd)/outro.pdf:/app/document.pdf"   ou aponte PDF_PATH.
COPY --chown=app:app document.pdf ./

USER app

# Sem HEALTHCHECK: a imagem é uma CLI interativa de execução pontual,
# não um serviço de longa duração com estado de saúde a monitorar.

# Ingestão: docker run --rm <img> python src/ingest.py
CMD ["python", "src/chat.py"]
