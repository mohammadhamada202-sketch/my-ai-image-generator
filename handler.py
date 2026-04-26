import os
import torch
import runpod
import base64
import logging
import io
import gc  # مكتبة جمع القمامة البرمجية لتنظيف الذاكرة

# 1. تنظيف الذاكرة المؤقتة فور تشغيل الكود
def clear_system_cache():
    # مسح أي ملفات قديمة مخزنة في الذاكرة العشوائية (RAM) أو كرت الشاشة (GPU)
    torch.cuda.empty_cache()
    gc.collect()
    print("--- SYSTEM CACHE CLEARED: Ready for Fresh Connection ---")

# استدعاء التنظيف فور إقلاع السيرفر
clear_system_cache()

from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعداد السجلات
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. إعداد OpenAI (تأكد من وضع المفتاح هنا)
client = OpenAI(api_key="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx")

# 3. تحميل الموديل (SDXL)
logger.info("Loading Fresh SDXL Pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def translate_and_enhance(prompt):
    """ التواصل المباشر مع OpenAI لضمان الترجمة """
    try:
        logger.info(f"--- TRIGGERING OPENAI FOR: {prompt} ---")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to English and enhance the prompt for SDXL. Return ONLY the English text."},
                {"role": "user", "content": prompt}
            ],
            timeout=15
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"--- OPENAI RESPONSE RECEIVED: {result} ---")
        return result
    except Exception as e:
        logger.error(f"--- OPENAI CONNECTION FAILED: {str(e)} ---")
        return prompt

def handler(job):
    # تنظيف الذاكرة قبل كل عملية رسم جديدة لضمان عدم وجود تداخل
    torch.cuda.empty_cache()
    
    try:
        job_input = job['input']
        user_prompt = job_input.get('prompt', '')
        
        # تنفيذ الترجمة الإجبارية عبر OpenAI
        final_prompt = translate_and_enhance(user_prompt)
        
        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                num_inference_steps=30,
                width=job_input.get('width', 1024),
                height=job_input.get('height', 1024)
            ).images[0]

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return {"image_base64": base64.b64encode(buffered.getvalue()).decode("utf-8")}
        
    except Exception as e:
        logger.error(f"Handler Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
