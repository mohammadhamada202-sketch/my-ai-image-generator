import os
# منع تضارب نسخ نيمباي وتفعيل الاستقرار
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
import torch
import runpod
import base64
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- إعدادات OpenAI ---
# تأكد من وضع مفتاحك الجديد هنا
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx" 
client = OpenAI(api_key=OPENAI_API_KEY)

# ستايلات إضافية لتحسين الجودة
STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k, raw photo, masterpiece, sharp focus",
    "anime": "anime style art, vibrant colors, studio ghibli, high quality",
    "cinematic": "cinematic movie still, moody lighting, epic composition",
    "cartoon": "cute 3d cartoon style, pixar aesthetics, vibrant colors"
}

def enhance_and_translate(user_input):
    """
    هذه الدالة تقوم بترجمة النص للعربية وتحويله لبرومبت احترافي
    """
    try:
        logger.info(f"Translating input: {user_input}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Translate the user input to English and expand it into a highly detailed visual prompt for Stable Diffusion. Return ONLY the English prompt."
                },
                {"role": "user", "content": user_input}
            ],
            timeout=10
        )
        enhanced = response.choices[0].message.content.strip()
        logger.info(f"Final Prompt after AI: {enhanced}")
        return enhanced
    except Exception as e:
        logger.error(f"AI Translation failed: {e}")
        return user_input # العودة للنص الأصلي كخيار أمان

# --- تحميل الموديل ---
logger.info("Loading SDXL Pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def handler(job):
    try:
        job_input = job['input']
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        
        # الخطوة الحاسمة: ترجمة وتحسين النص قبل الرسم
        translated_prompt = enhance_and_translate(user_prompt)
        
        # دمج النص المترجم مع الستايل المختار
        final_prompt = f"{translated_prompt}, {STYLE_MODIFIERS.get(style, '')}"

        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                num_inference_steps=30,
                width=job_input.get('width', 1024),
                height=job_input.get('height', 1024)
            ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
