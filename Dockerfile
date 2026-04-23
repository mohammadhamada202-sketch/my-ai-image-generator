# استخدام صورة بايثون مع تعريفات CUDA جاهزة من RunPod
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# تحديد مجلد العمل في الجذر
WORKDIR /

# نسخ ملف المكتبات أولاً
COPY requirements.txt .

# نسخ ملف الـ handler البرمجي
COPY handler.py .

# تحديث pip وتثبيت المكتبات ثم تشغيل الملف البرمجي
# تم دمجهم في أمر واحد لضمان التنفيذ عند بدء تشغيل الـ Pod
CMD ["sh", "-c", "pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt && python3 -u /handler.py"]
