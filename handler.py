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
        # تأكد أنك وضعت المفتاح الجديد الذي يبدأ بـ AQ. في RunPod
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # استخدام v1 هو الصحيح للمفاتيح الاحترافية
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        user_text = job_input.get('prompt', '')

        # الترجمة والتحسين (OpenAI)
        final_optimized_prompt = translate_and_optimize(user_text)
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # الحل الجذري لـ 404: تجربة المسميات المتوافقة مع Google Cloud v1
            potential_models = ['gemini-1.5-flash-002', 'gemini-1.5-flash']
            
            response = None
            for model_id in potential_models:
                try:
                    print(f"Trying to connect with: {model_id}")
                    response = client.models.generate_content(
                        model=model_id, 
                        contents=f"Generate a cinematic, hyper-realistic 8k image: {final_optimized_prompt}. Aspect ratio {width}:{height}"
                    )
                    if response: break
                except Exception as e:
                    print(f"Model {model_id} failed: {str(e)}")
                    continue
            
            if not response:
                raise Exception("Could not connect to any Gemini models. Please check API activation.")

            # استخراج الصورة
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
        print(f"--- [FINAL HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
