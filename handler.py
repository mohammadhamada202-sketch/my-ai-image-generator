import runpod
import torch
import base64
import os
import gc
from io import BytesIO
from PIL import Image
from openai import OpenAI
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

# إعدادات المجلدات
MODEL_CACHE_DIR = "/workspace/models"

# 1. إعداد OpenAI (تأكد من وجود المفتاح في Environment Variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. تحميل الموديل خارج الـ handler لضمان السرعة ومنع تسريب الذاكرة
try:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0", 
        torch_dtype=torch.float16, 
        variant="fp16",
        cache_dir=MODEL_CACHE_DIR
    ).to("cuda")
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    print("--- Neural Engine Loaded Successfully ---")
except Exception as e:
    print(f"Error loading model: {str(e)}")

def enhance_prompt(prompt):
    """تحسين البرومبت باستخدام GPT"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional AI prompt engineer. Enhance the user input for SDXL. Return ONLY the English prompt."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT Enhancement failed: {str(e)}")
        return prompt

def handler(job):
    try:
        # تفريغ الذاكرة المؤقتة لكرت الشاشة قبل كل عملية
        torch.cuda.empty_cache()
        gc.collect()

        job_input = job['input']
        image_b64 = job_input.get('image')
        user_prompt = job_input.get('prompt', '')
        strength = float(job_input.get('strength', 0.5))
        
        if not image_b64:
            return {"error": "No image data found in request"}

        # 1. تحسين النص
        final_prompt = enhance_prompt(user_prompt)

        # 2. معالجة الصورة الأصلية
        init_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
        init_image = init_image.resize((1024, 1024)) # توحيد المقاس للأداء المستقر

        # 3. توليد الصورة المعدلة
        output = pipe(
            prompt=final_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=30,
            guidance_scale=7.5,
            negative_prompt="low quality, blurry, distorted, extra fingers, bad anatomy"
        ).images[0]

        # 4. التشفير للرد
        buffered = BytesIO()
        output.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image_base64": img_str,
            "enhanced_prompt": final_prompt
        }

    except Exception as e:
        # طباعة الخطأ بالتفصيل في السجلات لتعرف ماذا حدث
        print(f"CRITICAL ERROR: {str(e)}")
        return {"error": str(e)}

# بدء السيرفر
runpod.serverless.start({"handler": handler})
