# استخدام نسخة مستقرة من RunPod
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# الخطوة الأهم: تحديث المكتبات الأساسية معاً لضمان التوافق
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    huggingface-hub==0.23.0 \
    transformers==4.38.0 \
    accelerate==0.27.0

# تثبيت بقية المكتبات المطلوبة للمشروع
RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.26.3 \
    xformers==0.0.22.post7 \
    opencv-python-headless \
    pillow \
    openai

# نسخ ملفات الأكواد
COPY handler.py .
COPY video_engine.py .

CMD [ "python", "-u", "/handler.py" ]
