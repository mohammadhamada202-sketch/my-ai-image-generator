import runpod
import os
import google.generativeai as genai
from translator_helper import translate_and_optimize
from dimensions_helper import get_image_dimensions
from avatar_generator import generate_avatar

# إعداد مفتاح Gemini للتوليد
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)

def handler(job):
    try:
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # 1. استخدام OpenAI للترجمة والتحسين الاحترافي
        final_optimized_prompt = translate_and_optimize(user_text)
        
        # 2. جلب المقاسات
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # 3. التوليد عبر Gemini (Nano Banana 2)
            model = genai.GenerativeModel('gemini-3-flash-image')
            response = model.generate_content(
                f"{final_optimized_prompt}. Aspect ratio: {width}:{height}"
            )
            return response.candidates[0].content.parts[0].inline_data.data
        else:
            # وضع الأفاتار (Image-to-Image)
            image_b64 = job_input.get('image')
            return generate_avatar(image_b64, final_optimized_prompt, style)

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
