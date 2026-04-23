# استخدام صورة بايثون جاهزة ومجهزة للذكاء الاصطناعي
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# تحديد مجلد العمل
WORKDIR /

# نسخ ملف المكتبات وتثبيته
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كود البايثون
COPY handler.py .

# الأمر الذي سيشغل السيرفر فور الإقلاع
CMD [ "python3", "-u", "/handler.py" ]
