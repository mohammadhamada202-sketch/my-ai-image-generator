# استخدام النسخة الحالية المستقرة لديك
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. تنظيف أي نسخ قديمة ومنع تثبيت NumPy 2.0
RUN pip uninstall -y numpy xformers

# 2. تثبيت الإصدارات المتوافقة بترتيب دقيق
# نستخدم numpy<2 لضمان التوافق مع PyTorch و Diffusers
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir \
    huggingface-hub==0.23.2 \
    transformers==4.40.0 \
    accelerate==0.30.0 \
    diffusers==0.27.2 \
    openai \
    runpod

# 3. تثبيت xformers المتوافق مع CUDA 11.8
RUN pip install --no-cache-dir xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

COPY handler.py .
COPY video_engine.py .

CMD [ "python", "-u", "/handler.py" ]
