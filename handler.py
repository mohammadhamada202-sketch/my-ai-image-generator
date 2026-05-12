import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        # المفاتيح الاحترافية AQ في أوروبا تعمل بامتياز مع v1
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # 1. المترجم
        final_optimized_prompt = translate_and_optimize(user_text)

        # 2. الحل الجذري: استخدام الاسم التقني الكامل لـ Imagen 3
        # حسابات Cloud المدفوعة تتعرف على هذا المسمى مباشرة
        try:
            response = client.models.generate_content(
                model='imagen-3.0-generate-002', 
                contents=f"{final_optimized_prompt}, hyper-realistic photography, 8k, extreme detail"
            )
            image_bytes = response.candidates[0].content.parts[0].inline_data.data
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            # إذا استمر الـ 404، سنجرب المسمى العام المتوافق مع v1
            print(f"Direct path failed, trying fallback: {str(e)}")
            response = client.models.generate_content(
                model='gemini-1.5-flash-002', # إضافة -002 ضرورية أحياناً في v1
                contents=f"Generate an image: {final_optimized_prompt}"
            )
            image_bytes = response.candidates[0].content.parts[0].inline_data.data
            return base64.b64encode(image_bytes).decode("utf-8")

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": f"Model Access Issue. Detail: {str(e)}"}

runpod.serverless.start({"handler": handler})
