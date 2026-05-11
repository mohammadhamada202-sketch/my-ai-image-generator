import runpod
import os
import base64
import io
from google import genai  # المكتبة الجديدة التي ثبتناها
from translator_helper import translate_and_optimize
from dimensions_config import get_image_dimensions # تصحيح اسم الملف
from avatar_generator import generate_avatar

def handler(job):
    try:
        # إعداد عميل Gemini باستخدام المفتاح من إعدادات RunPod
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # 1. الترجمة والتحسين عبر OpenAI (تأكد من وجود مفتاح OPENAI_API_KEY)
        final_optimized_prompt = translate_and_optimize(user_text)
        
        # 2. جلب المقاسات من الملف المحلي
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # --- وضع توليد الصور من النص ---
            response = client.models.generate_image(
                model='gemini-3-flash-image',
                prompt=final_optimized_prompt,
                config={'aspect_ratio': f"{width}:{height}"}
            )
            # إرجاع الصورة بصيغة Base64 للموقع
            return base64.b64encode(response.image.bits).decode("utf-8")
            
        else:
            # --- وضع الأفاتار (تحويل صورة لصورة) ---
            image_b64 = job_input.get('image')
            # استدعاء دالة الأفاتار المطورة التي تستخدم Gemini API
            output_img = generate_avatar(image_b64, final_optimized_prompt, style)
            
            # تحويل صورة PIL الناتجة إلى Base64 لإرسالها للموقع
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"--- [HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر على RunPod
runpod.serverless.start({"handler": handler})
