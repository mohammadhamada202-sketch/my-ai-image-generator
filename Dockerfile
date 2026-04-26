FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. تحديث pip وتثبيت numpy بشكل قسري ومنفرد في البداية
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --force-reinstall numpy==1.24.3

# 2. تثبيت المكتبات الأساسية (بترتيب يمنع تضارب الإصدارات)
RUN pip install --no-cache-dir \
    huggingface-hub==0.23.2 \
    transformers==4.40.0 \
    accelerate==0.30.0 \
    diffusers==0.27.2

# 3. تثبيت بقية متطلبات المشروع
RUN pip install --no-cache-dir \
    runpod \
    xformers==0.0.22.post7 \
    opencv-python-headless \
    pillow \
    openai

# نسخ الملفات
COPY handler.py .
COPY video_engine.py .

CMD [ "python", "-u", "/handler.py" ]
