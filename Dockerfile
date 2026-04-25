# استخدام نسخة مستقرة ومؤكدة الوجود
FROM runpod/pytorch:3.10-2.0.1-117-devel

WORKDIR /

# نسخ جميع الملفات
COPY . .

# تثبيت المكتبات (أضفت --no-cache-dir لتسريع وتقليل الأخطاء)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir runpod diffusers transformers accelerate xformers opencv-python-headless pillow openai

CMD [ "python", "-u", "/handler.py" ]
