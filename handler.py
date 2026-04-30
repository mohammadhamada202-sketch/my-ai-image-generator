import runpod
import torch
import base64
import os
import gc
from io import BytesIO
from PIL import Image
from openai import OpenAI
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

# --- تهيئة المحرك العصبوني ---
MODEL_ID = "SG161222/RealVisXL_V4.0"
MODEL_CACHE = "/workspace/models"

# إعداد OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# تحميل الموديل خارج الـ handler لضمان بقائه في الذاكرة
try:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        variant="fp16",
        cache_dir=MODEL_CACHE
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
    print(f"Loading Error: {str(e)}")

def handler(job):
    try:
        # تنظيف الذاكرة لضمان عدم حدوث Crash
        torch.cuda.empty_cache()
        gc.collect()

        job_input = job['input']
        mode = job_input.get('mode', 'text')
        user_prompt = job_input.get('prompt', '')
        image_b64 = job_input.get('image')

        # 1. تحسين البرومبت عبر GPT (مع معالجة خطأ الرصيد)
        final_prompt = user_prompt
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Enhance this for AI art: {user_prompt}"}],
                max_tokens=100
            )
            final_prompt = res.choices[0].message.content.strip()
        except:
            print("OpenAI skip - using original prompt")

        # 2. المعالجة في حالة Image-to-Image
        if mode == 'img2img' or image_b64:
            init_img = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
            init_img = init_img.resize((1024, 1024))
            
            output = pipe(
                prompt=final_prompt,
                image=init_img,
                strength=float(job_input.get('strength', 0.5)),
                num_inference_steps=30
            ).images[0]
        
        # 3. المعالجة في حالة Text-to-Image
        else:
            # ملاحظة: الموديل محمل كـ Img2Img، للتحويل الكامل للنص نحتاج لإرسال صورة فارغة أو استخدام Pipeline مختلف. 
            # الأفضل للمشروع الحالي هو استخدام Img2Img دائماً لضمان الثبات.
            return {"error": "Please provide a source image for this model version"}

        # 4. التشفير والرد
        buf = BytesIO()
        output.save(buf, format="PNG")
        return {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
