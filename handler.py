import runpod
import os
import base64
import io
from google import genai
from google.genai import types 
from translator_helper import translate_and_optimize
from dimensions_config import get_image_dimensions
from avatar_generator import generate_avatar

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # [1] الترجمة والتحسين عبر OpenAI (تعمل بنجاح)
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Success! Translated Prompt: {final_optimized_prompt}")

        # [2] المقاسات
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # [3] وصف فائق الواقعية لضمان جودة "تُرى بالعين"
            ultra_hd_prompt = (
                f"{final_optimized_prompt}, hyper-realistic photography, 8k resolution, "
                "shot on 35mm lens, f/1.8, cinematic lighting, sharp focus, "
                "extreme detail, realistic skin textures, professional masterpiece"
            )

            # [4] التوليد عبر Imagen 3 باستخدام الدالة الأكثر استقراراً
            response = client.models.generate_content(
                model='imagen-3.0-generate-002', 
                contents=ultra_hd_prompt,
                config=types.GenerateContentConfig(
                    # تمرير المقاسات هنا لضمان قبولها
                    candidate_count=1,
                    # ملاحظة: بعض الإصدارات تتطلب دمج المقاسات في النص إذا لم يدعمها الـ Config
                )
            )
            
            # استخراج الصورة من Inline Data
            image_bytes = response.candidates[0].content.parts[0].inline_data.data
            return base64.b64encode(image_bytes).decode("utf-8")
            
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

runpod.serverless.start({"handler": handler})
