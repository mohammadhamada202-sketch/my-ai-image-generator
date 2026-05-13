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
        # جلب مفتاح الـ API الجديد من إعدادات RunPod
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # إعداد العميل (الافتراضي يعمل بشكل ممتاز مع مفاتيح AI Studio)
        client = genai.Client(api_key=api_key)
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # 1. ترجمة وتحسين النص (عبر OpenAI) لضمان فهم Gemini العميق للمشهد
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Optimized Prompt: {final_optimized_prompt}")

        # 2. جلب المقاسات المطلوبة
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # 3. توليد الصورة باستخدام الموديل الذي أكدنا وجوده في حسابك
            # gemini-2.5-flash هو الأحدث ويدعم توليد الصور بدقة 8k
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=f"Generate a cinematic, hyper-realistic 8k photo: {final_optimized_prompt}. Aspect ratio {width}:{height}"
            )
            
            # استخراج بيانات الصورة وتحويلها لـ Base64
            image_bytes = response.candidates[0].content.parts[0].inline_data.data
            return base64.b64encode(image_bytes).decode("utf-8")
            
        else:
            # 4. مسار الأفاتار (تحويل الصور الشخصية)
            image_b64 = job_input.get('image')
            output_img = generate_avatar(image_b64, final_optimized_prompt, style)
            
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        # طباعة الخطأ في Logs الخاصة بـ RunPod لتسهيل تتبعه
        print(f"--- [CRITICAL HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر المخصص لـ RunPod
runpod.serverless.start({"handler": handler})
