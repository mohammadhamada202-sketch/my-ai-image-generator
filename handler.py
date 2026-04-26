import os
import torch
import runpod
import base64
import logging
import io
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# كود يضمن وجود المفتاح حتى لو فشل RunPod في قراءته
api_key = os.getenv("OPENAI_API_KEY", "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")

# التحقق من وجود المفتاح قبل تشغيل العميل
if not api_key or "sk-proj" not in api_key:
    raise ValueError("--- [CRITICAL] OpenAI API Key is missing or invalid! ---")

client = OpenAI(api_key=api_key)
# تعريف الكليينت
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger.info("Starting Fresh SDXL Pipeline in Romania...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", 
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

def handler(job):
    # فحص المفتاح عند كل طلب للتأكد
    check_key()
    
    job_input = job['input']
    user_prompt = job_input.get('prompt', '')
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Enhance this prompt: {user_prompt}"}],
            timeout=15
        )
        final_prompt = response.choices[0].message.content.strip()
        logger.info(f"--- OPENAI SUCCESS: {final_prompt} ---")
    except Exception as e:
        logger.error(f"--- OPENAI FAILED: {str(e)} ---")
        final_prompt = user_prompt

    with torch.inference_mode():
        image = pipe(prompt=final_prompt, num_inference_steps=30).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}

runpod.serverless.start({"handler": handler})
