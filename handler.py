import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        # المفاتيح الاحترافية تتطلب v1 دائماً
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # 1. الترجمة
        final_optimized_prompt = translate_and_optimize(user_text)

        # 2. الحل الجذري: استخدام دالة images.generate الحصرية للمفاتيح الاحترافية
        # هذا المسار لا يعطي 404 لأنه مخصص حصراً لموديلات Imagen
        try:
            response = client.images.generate(
                model='imagen-3.0-generate-002', 
                prompt=f"{final_optimized_prompt}, hyper-realistic, 8k, cinematic lighting",
                config={
                    'aspect_ratio': '1:1', # يمكنك تعديلها حسب الحاجة
                    'safety_filter_level': 'block_few'
                }
            )
            
            # في هذه الدالة، الصورة تكون في generated_images
            image_bytes = response.generated_images[0].image.bits
            return base64.b64encode(image_bytes).decode("utf-8")
            
        except Exception as e:
            print(f"Primary Path Failed: {str(e)}")
            # محاولة أخيرة باستخدام الاسم المختصر imagen-3
            response = client.images.generate(
                model='imagen-3',
                prompt=final_optimized_prompt
            )
            image_bytes = response.generated_images[0].image.bits
            return base64.b64encode(image_bytes).decode("utf-8")

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": f"Cloud API Access Error: {str(e)}"}

runpod.serverless.start({"handler": handler})
