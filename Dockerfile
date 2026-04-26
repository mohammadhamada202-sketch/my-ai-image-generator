# استخدام نسخة مستقرة من RunPod
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. تنظيف شامل لأي مخلفات قديمة
RUN pip uninstall -y transformers huggingface-hub accelerate diffusers numpy

# 2. تثبيت numpy أولاً بشكل منفصل لضمان توفره للمكتبات الأخرى
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3

# 3. تثبيت الحزمة المتوافقة "الذهبية" (إصدارات مترابطة برمجياً)
RUN pip install --no-cache-dir \
    huggingface-hub==0.23.2 \
    transformers==4.40.0 \
    accelerate==0.30.0 \
    diffusers==0.27.2

# 4. تثبيت المكتبات التقنية المتبقية
RUN pip install --no-cache-dir \
    runpod \
    xformers==0.0.22.post7 \
    opencv-python-headless \
    pillow \
    openai

# نسخ الملفات
COPY handler.py .
COPY video_engine.py .

# تشغيل السيرفر
CMD [ "python", "-u", "/handler.py" ]
