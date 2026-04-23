FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /
# نسخ الملفات فقط دون تثبيتها الآن لتجنب فشل الـ Build
COPY requirements.txt .
COPY handler.py .

# سنقوم بالتثبيت عند الإقلاع (Runtime)
CMD ["sh", "-c", "pip install --no-cache-dir -r requirements.txt && python3 -u /handler.py"]
