FROM python:3.11-slim AS builder

ARG INSTALL_THEME_ML=false

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY backend/requirements-runtime.txt backend/requirements-theme-ml.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-runtime.txt && \
    if [ "$INSTALL_THEME_ML" = "true" ]; then pip install --no-cache-dir -r requirements-theme-ml.txt; fi

COPY backend/ /build/backend

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r -g 1000 stockscanner && \
    useradd -r -u 1000 -g stockscanner stockscanner

ENV HOME=/app/data
ENV XDG_CACHE_HOME=/app/data/.cache
ENV PATH=/opt/venv/bin:$PATH

COPY --from=builder /opt/venv /opt/venv
COPY --chown=stockscanner:stockscanner backend/ .

RUN mkdir -p /app/data/.cache && \
    chown -R stockscanner:stockscanner /app/data

USER stockscanner

EXPOSE 8000

# WEB_CONCURRENCY is uvicorn's native worker-count env var. Each worker opens
# its own DB pool (DB_POOL_SIZE + DB_MAX_OVERFLOW connections); lower on small hosts.
ENV WEB_CONCURRENCY=4

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
