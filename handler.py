import runpod
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from googletrans import Translator
import base64
from io import BytesIO
import os

translator = Translator()

# هذا هو المسار داخل الـ Network Volume
MODEL_CACHE_DIR = "/workspace/models"

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        
        print(f"--- [START] جاري العمل على: {prompt} ---")

        # 1. الترجمة
        try:
            prompt = translator.translate(prompt, dest='en').text
        except: pass

        # 2. تحميل الموديل (سيتحمل في الـ Volume مرة واحدة فقط)
        print(f"--- جاري فحص الموديل في {MODEL_CACHE_DIR} ---")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True,
            cache_dir=MODEL_CACHE_DIR # هنا يتم الحفظ الدائم
        ).to("cuda")
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()

        # 3. الرسم
        print("--- بدأت عملية الرسم... ---")
        image = pipe(prompt=prompt, num_inference_steps=30).images[0]

        # 4. التشفير
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        print(f"--- خطأ: {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
