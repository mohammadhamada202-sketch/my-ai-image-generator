FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

# نسخ ملف المكتبات والكود إلى السيرفر
COPY requirements.txt .
COPY handler.py .

# تثبيت المكتبات وتشغيل المحرك عند الإقلاع
CMD ["sh", "-c", "pip install --no-cache-dir -r requirements.txt && python3 -u /handler.py"]
