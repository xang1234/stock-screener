FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# Using UID 1000 to match typical host user for volume permissions
RUN groupadd -r -g 1000 stockscanner && \
    useradd -r -u 1000 -g stockscanner stockscanner

# Set cache directory to writable volume location
# This fixes yfinance TzCache/CookieCache permission errors
ENV HOME=/app/data
ENV XDG_CACHE_HOME=/app/data/.cache
# Keep Playwright browser binaries outside mounted /app/data volume.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Copy requirements first for better caching
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and xui-reader source trees.
# Build context must be repository root for these paths to resolve.
COPY --chown=stockscanner:stockscanner backend/ .
COPY --chown=stockscanner:stockscanner xui-reader ./xui-reader

# Install xui-reader package and Playwright Chromium browser.
RUN pip install --no-cache-dir -e ./xui-reader && \
    python -m playwright install --with-deps chromium

# Create data and cache directories with correct ownership
RUN mkdir -p /app/data/.cache /ms-playwright && \
    chown -R stockscanner:stockscanner /app/data /ms-playwright

# Switch to non-root user
USER stockscanner

# Expose port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/readyz || exit 1

# Default command (overridden in docker-compose for Celery workers)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
