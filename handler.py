import runpod
import os
import torch
from openai import OpenAI
from image_engine import ImageEngine # استدعاء المحرك من الملف الثاني

# --- الإعدادات ---
MODEL_CACHE_DIR = "/workspace/models"
MODEL_NAME = "SG161222/RealVisXL_V4.0"

# إعداد OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# بدء تشغيل المحرك (يتم تحميل الموديل مرة واحدة فقط عند إقلاع السيرفر)
engine = ImageEngine(MODEL_NAME, MODEL_CACHE_DIR)

def enhance_prompt_with_gpt(user_input, is_magic=False):
    """تحسين البرومبت وترجمته"""
    magic_context = "Suggest a cinematic and creative enhancement." if is_magic else ""
    system_prompt = (
        "You are an AI prompt engineer. Translate to English if needed and add professional visual details. "
        "Return ONLY the final prompt text."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{magic_context} User Input: {user_input}"}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except:
        return user_input

def handler(job):
    try:
        job_input = job['input']
        
        # استلام البيانات من الطلب القادم من الموقع
        image_64 = job_input.get('image')           # الصورة Base64
        raw_prompt = job_input.get('prompt', '')     # نص المستخدم
        style = job_input.get('style', 'realistic') # نوع الستايل
        ui_strength = job_input.get('strength', 50) # القيمة من 1 لـ 100
        is_magic = job_input.get('magic', False)    # هل زر السحر مفعل؟

        if not image_64:
            return {"error": "Image data is required"}

        # 1. تحسين البرومبت عبر GPT
        final_prompt = enhance_prompt_with_gpt(raw_prompt, is_magic)

        # 2. استدعاء المحرك لمعالجة الصورة (Image-to-Image)
        result_base64 = engine.process(image_64, final_prompt, style, ui_strength)

        # 3. إرجاع النتيجة النهائية
        return {
            "image_base64": result_base64,
            "enhanced_prompt": final_prompt,
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e)}

# بدء خدمة RunPod
runpod.serverless.start({"handler": handler})
