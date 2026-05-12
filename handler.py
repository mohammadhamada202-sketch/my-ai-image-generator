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
        api_key = os.environ.get("GEMINI_API_KEY")
        # نستخدم v1 لضمان استقرار المشاريع المدفوعة في ألمانيا
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        user_text = job_input.get('prompt', '')

        # 1. المترجم
        final_optimized_prompt = translate_and_optimize(user_text)
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # الحل النهائي: استدعاء Imagen 3 مباشرةً بدون وسيط Gemini
            # هذا المسار هو الوحيد الذي يعمل 100% مع حسابات Cloud المدفوعة
            try:
                response = client.models.generate_content(
                    model='imagen-3.0-generate-002', 
                    contents=f"{final_optimized_prompt}, hyper-realistic photography, 8k, extreme detail. Aspect ratio {width}:{height}"
                )
                image_bytes = response.candidates[0].content.parts[0].inline_data.data
                return base64.b64encode(image_bytes).decode("utf-8")
            except Exception as e:
                # إذا فشل الأول، نجرب النسخة "المبسطة" التي تحبها سيرفرات أوروبا
                print(f"Fallback to simple model name... Error was: {str(e)}")
                response = client.models.generate_content(
                    model='imagen-3.0-capability-001', # موديل مخصص للحسابات التجارية
                    contents=f"Generate image: {final_optimized_prompt}"
                )
                image_bytes = response.candidates[0].content.parts[0].inline_data.data
                return base64.b64encode(image_bytes).decode("utf-8")
            
        else:
            # وضع الأفاتار
            image_b64 = job_input.get('image')
            output_img = generate_avatar(image_b64, final_optimized_prompt, job_input.get('style', 'photorealistic'))
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        # إذا استمر الخطأ، اطبع قائمة الموديلات المتاحة لك لتعرف ماذا تختار
        return {"error": f"Model Access Issue. Check if Imagen is enabled. Detail: {str(e)}"}

runpod.serverless.start({"handler": handler})
