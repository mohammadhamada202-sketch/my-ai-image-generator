# استخدم نسخة بايثون نحيفة بدلاً من Pytorch الضخم لتوفير المساحة والمال
FROM python:3.10-slim

WORKDIR /

# تثبيت الأدوات الأساسية فقط
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# تثبيت المكتبات المطلوبة فقط لنظام الـ API
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    google-generativeai \
    openai \
    Pillow \
    requests

# نسخ ملفات الكود الخاصة بك
COPY handler.py .
COPY avatar_generator.py .
COPY translator_helper.py .
COPY dimensions_config.py .

# ملاحظة: إذا كان ملف dimensions_config يسمى في الكود dimensions_helper، تأكد من مطابقة الاسم
# COPY dimensions_config.py ./dimensions_helper.py

CMD [ "python", "-u", "/handler.py" ]
