# استخدام نسخة بايثون أحدث (3.11) لتجنب التحذيرات
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# تعيين مجلد العمل
WORKDIR /

# تثبيت الأدوات الأساسية للنظام
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# تثبيت المكتبات البرمجية
# أضفنا supabase هنا لضمان وجودها في الحاوية
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    google-genai \
    openai \
    supabase \
    Pillow \
    requests

# نسخ ملفات المشروع
COPY handler.py .
COPY avatar_generator.py .
COPY translator_helper.py .
COPY dimensions_config.py .

# تشغيل السيرفر
CMD [ "python", "-u", "/handler.py" ]
