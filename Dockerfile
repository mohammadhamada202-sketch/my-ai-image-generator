# استخدام نسخة مستقرة من RunPod
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. مسح أي نسخ قديمة قد تكون موجودة مسبقاً في الصورة الأساسية
RUN pip uninstall -y transformers huggingface-hub accelerate diffusers

# 2. تثبيت الحزمة المتوافقة "الذهبية" (هذه النسخ تعمل مع بعضها بدون أخطاء Shards)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    huggingface-hub==0.23.2 \
    transformers==4.40.0 \
    accelerate==0.30.0 \
    diffusers==0.27.2

# 3. تثبيت بقية المكتبات التقنية للمشروع
RUN pip install --no-cache-dir \
    runpod \
    xformers==0.0.22.post7 \
    opencv-python-headless \
    pillow \
    openai

# نسخ الملفات (تأكد أن الأسماء مطابقة تماماً لما في جهازك)
COPY handler.py .
COPY video_engine.py .

# تشغيل السيرفر
CMD [ "python", "-u", "/handler.py" ]
