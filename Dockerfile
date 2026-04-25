# استخدام نسخة مستقرة جداً وتوافقية مع التعريفات القديمة والجديدة
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# تثبيت المكتبات الأساسية مع تحديد إصدارات متوافقة
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.24.0 \
    transformers \
    accelerate \
    xformers==0.0.22.post7 \
    opencv-python-headless \
    pillow \
    openai

# نسخ ملفات الكود إلى الحاوية
COPY handler.py .
COPY video_engine.py .

# تشغيل السيرفر
CMD [ "python", "-u", "/handler.py" ]
