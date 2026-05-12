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
        # التأكد من وجود الـ API Key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY is missing in environment variables"}

        client = genai.Client(api_key=api_key)
        job_input = job['input']
        
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # ترجمة النص
        final_prompt = translate_and_optimize(user_text)

        if mode == 'text':
            # طلب التوليد - الطريقة المضمونة لـ 1.5 flash
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=f"Generate a hyper-realistic cinematic 8k image: {final_prompt}"
            )
            
            # استخراج الصورة
            image_bytes = response.candidates[0].content.parts[0].inline_data.data
            return base64.b64encode(image_bytes).decode("utf-8")
        else:
            # وضع الأفاتار
            image_b64 = job_input.get('image')
            output_img = generate_avatar(image_b64, final_prompt, style)
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        return {"error": str(e)}

# السطر الأهم لضمان عمل RunPod
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
