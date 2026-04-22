import runpod
import torch
from diffusers import StableDiffusionXLPipeline
import io
import base64

# تحميل الموديل عند بدء التشغيل
print("--- [START] جاري تحميل الموديل... ---")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")
print("--- [READY] السيرفر جاهز تماماً لاستلام الطلبات ---")

def handler(job):
    try:
        prompt = job['input'].get('prompt', 'A high quality photo')
        print(f"--- جاري رسم: {prompt} ---")
        
        image = pipe(prompt=prompt, num_inference_steps=20).images[0]
        
        buffer = io.BytesIO()
        image.save(buffer, format="WebP")
        return {"image": base64.b64encode(buffer.getvalue()).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

# تشغيل السيرفر
runpod.serverless.start({"handler": handler})
