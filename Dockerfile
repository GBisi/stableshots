FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    QSIMBENCH_CACHE_DIR=/app/.qsimbench_cache

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY demo ./demo

RUN uv pip install --system ".[demo]"

EXPOSE 8501

CMD ["stableshot", "demo", "--host", "0.0.0.0", "--port", "8501"]
