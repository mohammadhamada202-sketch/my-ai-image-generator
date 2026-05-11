# استخدام نسخة بايثون أحدث (3.11) لتجنب التحذيرات
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    google-genai \
    openai \
    Pillow \
    requests

COPY handler.py .
COPY avatar_generator.py .
COPY translator_helper.py .
COPY dimensions_config.py .

CMD [ "python", "-u", "/handler.py" ]
