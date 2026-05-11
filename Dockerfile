# استخدام نسخة بايثون رسمية ومستقرة
FROM python:3.10-slim

# تجنب توقف البناء لطلب مدخلات من المستخدم
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

# تحديث النظام وتثبيت الأدوات الأساسية فقط مع محاولات إعادة الاتصال
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# تثبيت المكتبات المطلوبة للتعامل مع الـ APIs والصور
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    google-generativeai \
    openai \
    Pillow \
    requests

# نسخ ملفات الكود الخاصة بك
# تأكد أن أسماء الملفات في المجلد عندك تطابق هذه الأسماء تماماً
COPY handler.py .
COPY avatar_generator.py .
COPY translator_helper.py .
COPY dimensions_config.py .

# تشغيل الـ Handler
CMD [ "python", "-u", "/handler.py" ]
