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
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # الترجمة والتحسين (OpenAI)
        final_optimized_prompt = translate_and_optimize(user_text)
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # مصفوفة الموديلات الممكنة لتجاوز خطأ 404
            potential_models = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash-latest']
            
            response = None
            error_log = []

            for model_name in potential_models:
                try:
                    print(f"Trying to generate with: {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=f"Generate a cinematic, hyper-realistic 8k image: {final_optimized_prompt}. Aspect ratio {width}:{height}"
                    )
                    if response:
                        print(f"SUCCESS with model: {model_name}")
                        break
                except Exception as e:
                    error_log.append(f"{model_name}: {str(e)}")
                    continue

            if not response:
                raise Exception(f"All models failed. Details: {'; '.join(error_log)}")

            # استخراج الصورة
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

runpod.serverless.start({"handler": handler})
