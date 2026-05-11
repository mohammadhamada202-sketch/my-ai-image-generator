import runpod
import os
import base64
import io
from google import genai  # استدعاء المكتبة الجديدة 
from translator_helper import translate_and_optimize
from dimensions_config import get_image_dimensions # التأكد من تطابق اسم الملف [cite: 15]
from avatar_generator import generate_avatar

def handler(job):
    try:
        # إعداد العميل الجديد (Client) لمكتبة قوقل الحديثة 
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # 1. الترجمة والتحسين الاحترافي عبر OpenAI
        final_optimized_prompt = translate_and_optimize(user_text)
        
        # 2. جلب المقاسات الصحيحة
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # --- وضع توليد الصور من النص باستخدام Nano Banana 2 ---
            response = client.models.generate_image(
                model='gemini-3-flash-image',
                prompt=final_optimized_prompt,
                config={'aspect_ratio': f"{width}:{height}"}
            )
            # تحويل النتيجة لـ Base64 لإرسالها لموقعك
            return base64.b64encode(response.image.bits).decode("utf-8")
            
        else:
            # --- وضع الأفاتار (تحويل الصورة بذكاء Gemini) ---
            image_b64 = job_input.get('image')
            # استدعاء دالة الأفاتار المطورة
            output_img = generate_avatar(image_b64, final_optimized_prompt, style)
            
            # تحويل الصورة (PIL Image) لـ Base64
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"--- [HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

# بدء السيرفر
runpod.serverless.start({"handler": handler})
