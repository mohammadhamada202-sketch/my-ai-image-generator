# استخدام نسخة بايثون أحدث (3.11) لتجنب التحذيرات
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# تعيين مجلد العمل في الجذر الرئيسي
WORKDIR /

# تثبيت الأدوات الأساسية للنظام
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# تثبيت المكتبات البرمجية المطلوبة لمشروعك
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    google-genai \
    openai \
    supabase \
    Pillow \
    requests

# الحـــل ههنا: نسخ جميع ملفات المشروع دفعة واحدة (بما فيها ملفات الأنماط والمقاسات)
COPY . .

# تشغيل السيرفر
CMD [ "python", "-u", "/handler.py" ]
