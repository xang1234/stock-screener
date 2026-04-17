FROM python:3.11-slim AS builder

ARG INSTALL_THEME_ML=false
ARG INSTALL_XUI=false

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
COPY xui-reader /build/xui-reader
RUN if [ "$INSTALL_XUI" = "true" ]; then pip install --no-cache-dir -e /build/xui-reader; fi

FROM python:3.11-slim

ARG INSTALL_XUI=false
ARG INSTALL_PLAYWRIGHT=false

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r -g 1000 stockscanner && \
    useradd -r -u 1000 -g stockscanner stockscanner

ENV HOME=/app/data
ENV XDG_CACHE_HOME=/app/data/.cache
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV PATH=/opt/venv/bin:$PATH

COPY --from=builder /opt/venv /opt/venv
COPY --chown=stockscanner:stockscanner backend/ .
COPY --chown=stockscanner:stockscanner xui-reader ./xui-reader

RUN if [ "$INSTALL_XUI" = "true" ] && [ "$INSTALL_PLAYWRIGHT" = "true" ]; then python -m playwright install --with-deps chromium; fi && \
    mkdir -p /app/data/.cache /ms-playwright && \
    chown -R stockscanner:stockscanner /app/data /ms-playwright /app/xui-reader

USER stockscanner

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
