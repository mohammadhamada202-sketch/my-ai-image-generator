import runpod
import torch
import base64
from io import BytesIO
import os

# حل مشكلات التوافق
if not hasattr(torch, 'xpu'):
    torch.xpu = type('XPU', (), {'is_available': lambda: False, 'empty_cache': lambda: None})

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

MODEL_CACHE_DIR = "/workspace/models"

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get('prompt', '')
        
        # إعدادات الجودة العالية
        steps = 40 # زدت الخطوات لزيادة الدقة
        high_noise_frac = 0.8 # متى يبدأ المحسن عمله

        print(f"--- [START] توليد جودة خرافية لـ: {prompt} ---")

        # 1. الموديل الأساسي
        base = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)
        base.enable_xformers_memory_efficient_attention()

        # 2. الموديل المحسن (Refiner) - سيتم تحميله وتخزينه في الـ Volume
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        refiner.enable_xformers_memory_efficient_attention()

        # 3. عملية الرسم المزدوجة (Base + Refiner)
        print("--- المرحلة 1: الرسم الأساسي ---")
        image = base(
            prompt=prompt,
            num_inference_steps=steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        print("--- المرحلة 2: تحسين التفاصيل (Refining) ---")
        image = refiner(
            prompt=prompt,
            num_inference_steps=steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        # 4. التشفير
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
