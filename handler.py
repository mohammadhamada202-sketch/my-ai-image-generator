import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize
from dimensions_config import get_image_dimensions
from avatar_generator import generate_avatar

def handler(job):
    try:
        # 1. جلب المفتاح المفعّل (N5UI) من إعدادات RunPod
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # 2. إعداد العميل مع تحديد النسخة v1 لضمان التوافق مع موديلات 2.5
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # 3. تحسين النص وترجمته لضمان فهم Gemini للتفاصيل السينمائية
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Optimized Prompt: {final_optimized_prompt}")

        # 4. جلب مقاسات الصورة
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # 5. التوليد باستخدام الموديل الذي أكدنا وجوده في حسابك (Gemini 2.5 Flash)
            # هذا الموديل في عام 2026 هو المحرك الأساسي لتوليد الصور 8k
            print("Requesting image generation from gemini-2.5-flash...")
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=f"Generate a cinematic, hyper-realistic 8k photo: {final_optimized_prompt}. Aspect ratio {width}:{height}"
            )
            
            # استخراج الصورة وتحويلها لـ Base64
            image_bytes = response.candidates[0].content.parts[0].inline_data.data
            return base64.b64encode(image_bytes).decode("utf-8")
            
        else:
            # 6. مسار تحويل الصور الشخصية (الأفاتار)
            image_b64 = job_input.get('image')
            output_img = generate_avatar(image_b64, final_optimized_prompt, style)
            
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        # طباعة الخطأ التفصيلي في Logs الخاصة بـ RunPod
        print(f"--- [CRITICAL HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

# ربط الكود بـ RunPod Serverless
runpod.serverless.start({"handler": handler})
