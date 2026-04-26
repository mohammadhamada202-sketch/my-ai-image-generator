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

# وظيفة للتأكد من المفتاح المستخدم بدون كشفه بالكامل
def check_key():
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        # سيطبع أول 6 وأخر 4 أحرف فقط للتأكد
        logger.info(f"--- [KEY CHECK] Using Key: {key[:6]}...{key[-4:]} ---")
    else:
        logger.error("--- [KEY CHECK] NO KEY FOUND IN ENVIRONMENT! ---")

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
