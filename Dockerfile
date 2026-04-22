# تغيير الصورة الأساسية لنسخة رسمية ومتاحة دائماً من بايثون مع كودا
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /

# تثبيت المكتبات اللازمة للنظام (مهم جداً لهذا النوع من الصور)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD [ "python", "-u", "/handler.py" ]
