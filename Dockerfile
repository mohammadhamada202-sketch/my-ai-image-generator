FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. تنظيف شامل لإصدارات NumPy و xformers المتضاربة
RUN pip uninstall -y numpy xformers

# 2. تثبيت numpy المتوافق (أقل من الإصدار 2) أولاً بشكل منفصل [cite: 7, 24, 38]
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2"


# 3. تثبيت المكتبات الأساسية (النسخة الذهبية المستقرة)
RUN pip install --no-cache-dir \
    huggingface-hub==0.23.2 \
    transformers==4.40.0 \
    accelerate==0.30.0 \
    diffusers==0.27.2 \
    openai \
    runpod

# 4. تثبيت xformers متوافق مع CUDA 11.8 الموجود في الحاوية 
RUN pip install --no-cache-dir xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

# نسخ ملفات الأكواد
COPY handler.py .
COPY video_engine.py .

CMD [ "python", "-u", "/handler.py" ]
