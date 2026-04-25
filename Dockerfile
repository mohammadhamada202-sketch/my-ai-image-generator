# استخدام النسخة المستقرة
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. تحديث pip وتثبيت الإصدارات المتوافقة من مكتبات Hugging Face أولاً
RUN pip install --upgrade pip && \
    pip install --no-cache-dir huggingface-hub==0.20.3 transformers accelerate

# 2. تثبيت بقية المكتبات مع تحديد إصدار diffusers المستقر
RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.25.0 \
    xformers==0.0.22.post7 \
    opencv-python-headless \
    pillow \
    openai

# نسخ الملفات
COPY handler.py .
COPY video_engine.py .

CMD [ "python", "-u", "/handler.py" ]
