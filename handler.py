import os
import torch
import runpod
import base64
import logging
import io
import gc
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعدادات الاستقرار
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ضع مفتاحك هنا ---
client = OpenAI(api_key="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")

# تحميل الموديل (SDXL)
logger.info("Loading Fresh SDXL Pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def translate_and_enhance(prompt):
    """ دالة التواصل مع OpenAI مع نظام كشف أعطال """
    try:
        logger.info(f"--- [DEBUG] CALLING OPENAI FOR: {prompt} ---")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to English and enhance as a professional SDXL prompt. Return only the result."},
                {"role": "user", "content": prompt}
            ],
            timeout=15
        )
        res = response.choices[0].message.content.strip()
        logger.info(f"--- [DEBUG] OPENAI SUCCESS: {res} ---")
        return res
    except Exception as e:
        # هذا السطر سيكشف لك في الـ Logs سبب فشل الـ 5 دولار
        logger.error(f"--- [CRITICAL] OPENAI FAILED! REASON: {str(e)} ---")
        return prompt

def handler(job):
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        job_input = job['input']
        user_prompt = job_input.get('prompt', '')
        
        # تنفيذ الترجمة
        final_prompt = translate_and_enhance(user_prompt)
        
        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                num_inference_steps=30,
                width=job_input.get('width', 1024),
                height=job_input.get('height', 1024)
            ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return {"image_base64": img_str}
    except Exception as e:
        logger.error(f"Handler Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
