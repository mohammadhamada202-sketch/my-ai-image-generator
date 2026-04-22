FROM runpod/worker-v1-cuda12.1

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD [ "python", "-u", "/handler.py" ]
