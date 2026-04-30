FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

# 1. تنظيف شامل لإصدارات NumPy و xformers المتضاربة
RUN pip uninstall -y numpy xformers

# 2. تثبيت numpy المتوافق (أقل من الإصدار 2) أولاً بشكل منفصل
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2"

# 3. تثبيت المكتبات الأساسية (بما فيها openai)
RUN pip install --no-cache-dir \
    huggingface-hub==0.23.2 \
    transformers==4.40.0 \
    accelerate==0.30.0 \
    diffusers==0.27.2 \
    openai \
    runpod

# 4. تثبيت xformers متوافق مع CUDA 11.8
RUN pip install --no-cache-dir xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

# 5. نسخ جميع ملفات المشروع الجديدة (تأكد أن الأسماء تطابق ملفاتك في GitHub)
COPY __init__.py .
COPY handler.py .
COPY styles_config.py .
COPY dimensions_helper.py .
COPY text_generator.py .
COPY avatar_generator.py .
COPY translator_helper.py .

# أمر التشغيل الأساسي
CMD [ "python", "-u", "/handler.py" ]
