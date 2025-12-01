FROM python:3.11-slim

ARG DENO_VERSION=2.0.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DENO_NO_UPDATE_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip ca-certificates ffmpeg git && \
    arch="$(uname -m)" && \
    case "$arch" in \
        x86_64) deno_arch="x86_64-unknown-linux-gnu" ;; \
        aarch64|arm64) deno_arch="aarch64-unknown-linux-gnu" ;; \
        *) echo "Unsupported architecture for Deno: $arch" >&2; exit 1 ;; \
    esac && \
    curl -fsSL "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-${deno_arch}.zip" -o /tmp/deno.zip && \
    unzip -q /tmp/deno.zip -d /tmp && \
    mv /tmp/deno /usr/local/bin/deno && \
    chmod +x /usr/local/bin/deno && \
    rm -rf /tmp/deno.zip && \
    rm -rf /var/lib/apt/lists/*

# populate cache; this works around HuggingFace egress restrictions
RUN curl -fsSL "https://ndurner.de/download/aileen3/aileen3-cache.zip" -o /tmp/aileen3-cache.zip && \
    unzip -q /tmp/aileen3-cache.zip -d /tmp && \
    mkdir -p /root/.cache && \
    mv /tmp/aileen3-cache /root/.cache/aileen3 && \
    rm -rf /tmp/aileen3-cache.zip

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN chmod +x scripts/start_combined.sh

ENV API_SERVER_HOST=0.0.0.0 \
    API_SERVER_PORT=8000 \
    GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860
EXPOSE 8000

CMD ["scripts/start_combined.sh"]
