FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash git build-essential \
    && rm -rf /var/lib/apt/lists/*

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
