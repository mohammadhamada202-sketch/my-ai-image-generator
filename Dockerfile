FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /

RUN pip uninstall -y numpy xformers
RUN pip install --upgrade pip && pip install --no-cache-dir "numpy<2"

# تثبيت المكتبات مع تحديد الإصدارات المطلوبة
RUN pip install --no-cache-dir \
    huggingface-hub==0.23.2 \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    diffusers>=0.29.0 \
    openai \
    runpod

RUN pip install --no-cache-dir xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

# نسخ جميع الملفات الجديدة (مهم جداً)
COPY handler.py .
COPY styles_config.py .
COPY dimensions_helper.py .
COPY text_generator.py .
COPY avatar_generator.py .
COPY translator_helper.py .

CMD [ "python", "-u", "/handler.py" ]
