FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# تثبيت المكتبات الضرورية فقط
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ ملف التشغيل
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
