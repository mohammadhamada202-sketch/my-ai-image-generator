import runpod
import torch
import base64
import os
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
# استيراد محرك الفيديو من الملف الثاني
from video_engine import generate_video_logic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_CACHE_DIR = "/workspace/models"

STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k uhd, masterpiece, highly detailed",
    "anime": "anime style, vibrant, digital art, studio ghibli",
    "cinematic": "cinematic movie still, moody lighting, 35mm",
    "pixar": "disney pixar 3d style, vibrant colors, detailed render"
}

def enhance_prompt_via_gpt(user_input):
    if not OPENAI_API_KEY: return user_input
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Convert to a detailed English visual prompt for SDXL."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except: return user_input

# تحميل موديل الصور (يتم تحميله مرة واحدة عند بدء السيرفر)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=MODEL_CACHE_DIR
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def handler(job):
    try:
        job_input = job['input']
        mode = job_input.get('mode', 'image')

        # توجيه الطلب لمحرك الفيديو إذا كان الوضع فيديو
        if mode == 'video':
            logger.info("Redirecting to Video Engine...")
            return generate_video_logic(job)

        # وضع إنشاء الصور (النسخة الأولى)
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        
        final_prompt = enhance_prompt_via_gpt(user_prompt) + ", " + STYLE_MODIFIERS.get(style, "")
        
        image = pipe(
            prompt=final_prompt,
            num_inference_steps=30,
            width=job_input.get('width', 1024),
            height=job_input.get('height', 1024)
        ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return {"image_base64": base64.b64encode(buffered.getvalue()).decode("utf-8")}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
