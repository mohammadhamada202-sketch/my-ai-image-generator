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
        # إعداد العميل - التأكد من جلب المفتاح من البيئة
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # [1] المترجم (OpenAI)
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Success! Translated Prompt: {final_optimized_prompt}")

        # [2] المقاسات
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # [3] استخدام الموديل الأكثر استقراراً في فئة الفلاش
            # جيميني سيفهم أنك تريد صورة لأننا نبدأ بكلمة "Generate a realistic image"
            response = client.models.generate_content(
                model='gemini-1.5-flash', 
                contents=f"Generate a cinematic, hyper-realistic 8k image: {final_optimized_prompt}. Aspect ratio {width}:{height}"
            )
            
            # [4] استخراج الصورة بأكثر طريقة آمنة
            try:
                image_bytes = response.candidates[0].content.parts[0].inline_data.data
                return base64.b64encode(image_bytes).decode("utf-8")
            except Exception as img_err:
                print(f"Image extraction failed: {str(img_err)}")
                raise Exception("The model did not return valid image data.")
            
        else:
            # [5] وضع الأفاتار
            image_b64 = job_input.get('image')
            output_img = generate_avatar(image_b64, final_optimized_prompt, style)
            
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"--- [HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

# بدء السيرفر في RunPod
runpod.serverless.start({"handler": handler})
