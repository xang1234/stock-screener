# syntax=docker/dockerfile:1.7

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

FROM python:3.11-slim AS runtime-base

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
ENV MARKET_CALENDAR_DATA_ROOT=/opt/stockscanner/market-calendars

COPY --from=builder /opt/venv /opt/venv
COPY --chown=stockscanner:stockscanner backend/data/market_calendars/ /opt/stockscanner/market-calendars/
COPY --chown=stockscanner:stockscanner backend/ .

RUN mkdir -p /app/data/.cache && \
    chown -R stockscanner:stockscanner /app/data

USER stockscanner

EXPOSE 8000

# WEB_CONCURRENCY is uvicorn's native worker-count env var. Each worker opens
# its own DB pool (DB_POOL_SIZE + DB_MAX_OVERFLOW connections); lower on small hosts.
ENV WEB_CONCURRENCY=4

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM builder AS social-xui-builder
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
ARG XUI_READER_REF
RUN test "${#XUI_READER_REF}" -eq 40 \
    && case "$XUI_READER_REF" in *[!0-9a-fA-F]*) exit 1 ;; *) true ;; esac
RUN --mount=type=ssh,required=true \
    --mount=type=secret,id=github_known_hosts,target=/root/.ssh/known_hosts,required=true \
    pip install --no-cache-dir \
    "xui-reader[cli] @ git+ssh://git@github.com/xang1234/xui.git@${XUI_READER_REF}"

FROM runtime-base AS social-xui
USER root
COPY --from=social-xui-builder /opt/venv /opt/venv
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/stockscanner/playwright
RUN python -m playwright install --with-deps chromium \
    && chmod -R a+rX /opt/stockscanner/playwright \
    && rm -rf /var/lib/apt/lists/*
USER stockscanner

FROM runtime-base AS public
