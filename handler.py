import runpod
import os
import base64
import io
from google import genai  # التأكد من تثبيت google-genai
from translator_helper import translate_and_optimize
from dimensions_config import get_image_dimensions
from avatar_generator import generate_avatar

def handler(job):
    try:
        # إعداد العميل
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # 1. الترجمة والتحسين عبر OpenAI
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Success! Translated Prompt: {final_optimized_prompt}") # للـ Logs
        
        # 2. المقاسات
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # ندمج المقاسات في النص لتجنب مشاكل الـ Validation في المكتبة
            full_prompt = f"{final_optimized_prompt} (Aspect Ratio {width}:{height})"
            
            # التوليد باستخدام Nano Banana 2
            response = client.models.generate_content(
                model='gemini-3-flash-image',
                contents=full_prompt
            )
            
            image_bytes = response.candidates[0].content.parts[0].inline_data.data
            return base64.b64encode(image_bytes).decode("utf-8")
            
        else:
            # وضع الأفاتار
            image_b64 = job_input.get('image')
            output_img = generate_avatar(image_b64, final_optimized_prompt, style)
            
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"--- [HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

# بدء السيرفر
runpod.serverless.start({"handler": handler})
