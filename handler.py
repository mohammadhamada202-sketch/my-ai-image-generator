import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import torch
import runpod
import base64
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ضع مفتاح OpenAI الخاص بك هنا ---
client = OpenAI(api_key="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") 

# تحميل الموديل (SDXL)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# ستايلات إضافية لتحسين النتيجة بناءً على الاختيار
STYLE_MAP = {
    "realistic": "photorealistic, 8k, raw photo, highly detailed",
    "cinematic": "cinematic movie still, moody lighting, epic composition",
    "anime": "anime style art, vibrant colors, studio ghibli aesthetic",
    "cartoon": "3d cartoon disney style, vibrant, cute",
    "pixar": "high-end 3d render, pixar movie aesthetics, soft lighting"
}

def translate_prompt(prompt):
    """ إرسال الطلب لـ OpenAI للترجمة والتحسين """
    try:
        logger.info(f"Connecting to OpenAI for: {prompt}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to English and expand into a professional SDXL visual prompt. Return only the English text."},
                {"role": "user", "content": prompt}
            ],
            timeout=15
        )
        translated = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Result: {translated}")
        return translated
    except Exception as e:
        logger.error(f"OpenAI Failed: {e}")
        return prompt # العودة للأصل في حال العطل

def handler(job):
    try:
        job_input = job['input']
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        
        # 1. الترجمة فوراً
        final_prompt = translate_prompt(user_prompt)
        
        # 2. دمج الستايل
        full_prompt = f"{final_prompt}, {STYLE_MAP.get(style, '')}"
        
        logger.info(f"Final Drawing Prompt: {full_prompt}")

        # 3. الرسم
        with torch.inference_mode():
            image = pipe(
                prompt=full_prompt,
                num_inference_steps=30,
                width=job_input.get('width', 1024),
                height=job_input.get('height', 1024)
            ).images[0]

        # 4. التشفير والارسال
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return {"image_base64": base64.b64encode(buffered.getvalue()).decode("utf-8")}
    
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
