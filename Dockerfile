# استخدام نسخة مستقرة وموجودة فعلياً
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel

WORKDIR /

# نسخ جميع ملفات المشروع (handler.py, video_engine.py, requirements.txt)
COPY . .

# تحديث pip وتثبيت المكتبات
RUN pip install --upgrade pip
RUN pip install runpod diffusers transformers accelerate xformers opencv-python-headless pillow openai

# تشغيل الـ Handler
CMD [ "python", "-u", "/handler.py" ]
