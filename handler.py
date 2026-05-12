import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        # المفاتيح الاحترافية تتطلب v1
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # 1. المترجم
        final_optimized_prompt = translate_and_optimize(user_text)

        # 2. الاستدعاء عبر images.generate (المسار الذي لا يعطي 404)
        try:
            print(f"Executing Image Generation for: {final_optimized_prompt}")
            response = client.images.generate(
                model='imagen-3.0-generate-002',
                prompt=f"{final_optimized_prompt}, cinematic, 8k, realistic",
                config={
                    'number_of_images': 1,
                    'aspect_ratio': '1:1',
                    'safety_filter_level': 'block_few',
                    'output_mime_type': 'image/png'
                }
            )
            
            # في هذه الدالة، يتم استخراج البيانات هكذا:
            image_bytes = response.generated_images[0].image.bits
            return base64.b64encode(image_bytes).decode("utf-8")

        except Exception as e:
            print(f"Imagen Path failed: {str(e)}")
            # محاولة أخيرة باستخدام gemini-2.0-flash كموديل نصي يولد صوراً
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=f"Generate a photo of: {final_optimized_prompt}"
            )
            image_bytes = response.candidates[0].content.parts[0].inline_data.data
            return base64.b64encode(image_bytes).decode("utf-8")

    except Exception as e:
        print(f"--- [HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
