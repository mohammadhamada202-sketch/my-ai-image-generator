FROM runpod/pytorch:2.1.0-py3.10-cuda12.1-devel
WORKDIR /
COPY . .
RUN pip install runpod diffusers transformers accelerate xformers
CMD [ "python", "-u", "/handler.py" ]
