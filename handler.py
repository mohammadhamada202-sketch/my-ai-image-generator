import os
import torch
import runpod
import base64
import logging
import io
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعداد السجلات لمراقبة العملية
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
# ... باقي المكتبات

# يحاول الكود قراءة المفتاح من RunPod، وإذا لم يجده يستخدم المفتاح المكتوب (مؤقتاً)
api_key = os.getenv("OPENAI_API_KEY", "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")

client = OpenAI(api_key=api_key)

# تحميل الموديل (SDXL)
logger.info("Starting Fresh SDXL Pipeline in Romania...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", 
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def handler(job):
    job_input = job['input']
    user_prompt = job_input.get('prompt', '')
    
    # --- محاولة التواصل مع OpenAI ---
    logger.info(f"--- Calling OpenAI API for prompt: {user_prompt} ---")
    try:
        # ملاحظة: استخدمنا نموذج gpt-4o-mini لأنه الأسرع والأرخص
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to English and enhance as a professional SDXL prompt. Return only the English text."},
                {"role": "user", "content": user_prompt}
            ],
            timeout=15
        )
        final_prompt = response.choices[0].message.content.strip()
        logger.info(f"--- OpenAI Successfully Answered: {final_prompt} ---")
    except Exception as e:
        # إذا ظهر هذا السطر في الـ Logs، انسخ الخطأ لي فوراً
        logger.error(f"--- OpenAI Connection Failed: {str(e)} ---")
        final_prompt = user_prompt

    # --- عملية الرسم ---
    with torch.inference_mode():
        image = pipe(
            prompt=final_prompt,
            num_inference_steps=30,
            width=job_input.get('width', 1024),
            height=job_input.get('height', 1024)
        ).images[0]

    # تحويل الصورة إلى Base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}

runpod.serverless.start({"handler": handler})
