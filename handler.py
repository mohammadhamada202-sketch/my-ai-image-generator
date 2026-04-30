# handler.py
import runpod
import torch
import base64
import gc
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

# استيراد الملفات الفرعية
from styles_config import STYLE_PROMPTS, NEGATIVE_PROMPT
from dimensions_helper import get_dimensions
from text_generator import generate_from_text
from avatar_generator import generate_avatar
from translator_helper import translate_and_optimize # الملف الجديد

# إعداد النماذج
MODEL_ID = "SG161222/RealVisXL_V4.0"
pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe).to("cuda")

def handler(job):
    try:
        torch.cuda.empty_cache()
        gc.collect()

        job_input = job['input']
        mode = job_input.get('mode', 'text')
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        
        # --- السحر هنا: ترجمة وتحسين البرومبت عبر OpenAI ---
        optimized_prompt = translate_and_optimize(user_prompt)
        
        style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS['realistic'])

        if mode == 'text':
            width, height = get_dimensions(job_input)
            output_img = generate_from_text(pipe, optimized_prompt, style_prompt, NEGATIVE_PROMPT, width, height)
        else:
            image_b64 = job_input.get('image')
            output_img = generate_avatar(img_pipe, image_b64, optimized_prompt, style_prompt, NEGATIVE_PROMPT)

        buffered = BytesIO()
        output_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str, "used_prompt": optimized_prompt}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
