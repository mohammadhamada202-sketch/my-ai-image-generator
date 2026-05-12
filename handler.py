import runpod
import os
import base64
import io
from google import genai
from google.genai import types  # ضروري للتحكم بإعدادات الصورة
from translator_helper import translate_and_optimize
from dimensions_config import get_image_dimensions
from avatar_generator import generate_avatar

def handler(job):
    try:
        # [1] إعداد العميل باستخدام مفتاح API الخاص بك
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # [2] الترجمة والتحسين عبر OpenAI (تعمل بنجاح تام [cite: 10])
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Success! Translated Prompt: {final_optimized_prompt}") # [cite: 11]

        # [3] جلب المقاسات من dimensions_config.py
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # [4] توليد صورة "سينمائية" من النص باستخدام Imagen 3
            # دمج الخصائص الواقعية مباشرة في الوصف لضمان أعلى دقة
            ultra_hd_prompt = (
                f"{final_optimized_prompt}, hyper-realistic photography, 8k resolution, "
                "cinematic lighting, sharp focus, extreme details, realistic textures, "
                "professional masterpiece"
            )

            # استخدام الدالة المخصصة للصور لضمان عدم حدوث خطأ 404 أو 429
            response = client.models.generate_image(
                model='imagen-3.0-generate-002', # الموديل الأقوى حالياً
                prompt=ultra_hd_prompt,
                config=types.GenerateImageConfig(
                    number_of_images=1,
                    aspect_ratio=f"{width}:{height}", # تطبيق المقاسات المختارة
                    output_mime_type='image/png',
                    add_watermark=False
                )
            )
            
            # استخراج الصورة (المكتبة تعيد كائن صورة PIL مباشرة)
            output_image = response.generated_images[0].image
            
            # تحويل الصورة لـ Base64 لإرسالها للموقع
            buffered = io.BytesIO()
            output_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
        else:
            # [5] وضع الأفاتار (تحويل صورة لصورة بذكاء فائق)
            image_b64 = job_input.get('image')
            # استدعاء دالة الأفاتار المطورة التي تستخدم Imagen 3 أيضاً
            output_img = generate_avatar(image_b64, final_optimized_prompt, style)
            
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        # [6] تسجيل الأخطاء بوضوح للتشخيص [cite: 19]
        print(f"--- [HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل العامل البرمجي (Worker)
runpod.serverless.start({"handler": handler})
