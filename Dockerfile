# استخدام صورة بايثون مجهزة للذكاء الاصطناعي
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

# 1. تحديث pip وتثبيت المكتبات من requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. الكود السحري: تحميل الموديل وحفظه محلياً أثناء بناء الصورة (Baking)
# هذا السطر سيجعل الـ Build يطول ولكنه سيوفر رصيدك لاحقاً
RUN python3 -c "from diffusers import StableDiffusionXLPipeline; import torch; StableDiffusionXLPipeline.from_pretrained('SG161222/RealVisXL_V4.0', torch_dtype=torch.float16, variant='fp16', use_safetensors=True)"

# 3. نسخ ملف الـ handler
COPY handler.py .

# 4. تشغيل السيرفر
CMD ["python3", "-u", "/handler.py"]
