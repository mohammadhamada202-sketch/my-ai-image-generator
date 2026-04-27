import runpod
import torch
import base64
import os
from io import BytesIO
from openai import OpenAI  # تأكد من إضافة openai في requirements.txt
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعدادات الموديل والمسار
MODEL_CACHE_DIR = "/workspace/models"

# 1. جلب مفتاح OpenAI من إعدادات RunPod
# تأكد انك سميت المتغير في RunPod باسم OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def enhance_prompt_with_gpt(user_input):
    """
    وظيفة لترجمة وتحسين البرومبت باستخدام GPT
    """
    system_prompt = (
        "You are a professional AI image prompt engineer. "
        "If the input is in Arabic, translate it to English. "
        "Then, enhance the prompt to be highly detailed for a realistic AI image generator. "
        "Add details about lighting, camera settings (8k, raw photo), and environment. "
        "Keep the core idea of the user but make it professional. "
        "Return ONLY the enhanced English prompt."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # أو gpt-3.5-turbo حسب ميزانيتك
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        enhanced_text = response.choices[0].message.content.strip()
        return enhanced_text
    except Exception as e:
        print(f"GPT Error: {e}")
        return user_input # في حال فشل OpenAI نستخدم النص الأصلي

def handler(job):
    try:
        job_input = job['input']
        raw_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        if not raw_prompt:
            return {"error": "No prompt provided"}

        # 2. تحسين البرومبت وترجمته عبر OpenAI
        print(f"Original Prompt: {raw_prompt}")
        final_enhanced_prompt = enhance_prompt_with_gpt(raw_prompt)
        print(f"Enhanced Prompt: {final_enhanced_prompt}")

        # 3. تحميل الموديل وتوليد الصورة
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # تفعيل تحسينات الذاكرة
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass

        # توليد الصورة بالبرومبت "المحسن"
        image = pipe(
            prompt=final_enhanced_prompt,
            negative_prompt="low quality, blurry, distorted, extra fingers, bad anatomy, text, watermark",
            num_inference_steps=35,
            guidance_scale=7.5,
            width=width,
            height=height
        ).images[0]

        # تحويل النتيجة لـ Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image_base64": img_str,
            "used_prompt": final_enhanced_prompt # نرسل البرومبت الجديد للموقع إذا أحببت عرضه
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
