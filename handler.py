import os
import sys

# رقعة إصلاح لخطأ huggingface_hub
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

os.environ["DIFFUSERS_NO_ADVICE_LOGS"] = "1"
os.environ["ACCELERATE_USE_CPU"] = "False"

import runpod
import torch
import io
import base64
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from deep_translator import GoogleTranslator

# 1. إعداد الجهاز والموديل
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "SG161222/RealVisXL_V4.0"

try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    translator = GoogleTranslator(source='auto', target='en')
    print("✅ السيرفر جاهز تماماً")
except Exception as e:
    print(f"❌ خطأ أثناء التحميل: {e}")

# (بقية الكود الخاص بالـ STYLES والـ handler كما هو في الرد السابق...)

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get('prompt', 'A beautiful landscape')
        style = job_input.get('style', 'realistic')
        orientation = job_input.get('orientation', 'square')
        quality = job_input.get('quality', 'HD')

        en_prompt = translator.translate(prompt)
        
        # ستايلات محسنة
        styles_map = {
            "realistic": "photorealistic, 8k, raw photo, highly detailed",
            "anime": "anime art, high quality, vibrant",
            "digital_art": "concept art, digital painting, artstation",
            "3d_render": "unreal engine 5, octane render, 4k"
        }
        
        final_prompt = f"{en_prompt}, {styles_map.get(style, styles_map['realistic'])}"
        
        dims = {"portrait": (832, 1216), "landscape": (1216, 832), "square": (1024, 1024)}
        width, height = dims.get(orientation, (1024, 1024))
        steps = 35 if quality == "4K" else 25

        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
                guidance_scale=7.5
            ).images[0]
        
        buf = io.BytesIO()
        image.save(buf, format='WebP', quality=95)
        return {"status": "success", "image": base64.b64encode(buf.getvalue()).decode('utf-8')}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
