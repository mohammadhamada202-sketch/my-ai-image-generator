import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        # استخدام v1 كما أكدت التجربة الناجحة سابقاً
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')
        final_optimized_prompt = translate_and_optimize(user_text)

        print(f"Requesting from gemini-2.5-flash with prompt: {final_optimized_prompt}")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f"Generate a cinematic 8k photo: {final_optimized_prompt}"
        )

        # --- بداية المعالجة الصارمة للرد ---
        if not response or not response.candidates:
            raise Exception("Empty response from model")

        # محاولة استخراج الصورة من عدة مسارات محتملة في إصدار 2.5
        image_bytes = None
        first_part = response.candidates[0].content.parts[0]

        # المسار 1: inline_data.data (القياسي)
        if hasattr(first_part, 'inline_data') and first_part.inline_data:
            image_bytes = first_part.inline_data.data
        # المسار 2: الوصول المباشر لـ data (في بعض تحديثات المكتبة)
        elif hasattr(first_part, 'data'):
            image_bytes = first_part.data
        # المسار 3: فحص إذا كان هناك blob (في حال تغير البروتوكول)
        elif hasattr(first_part, 'blob'):
            image_bytes = first_part.blob.data

        if image_bytes:
            print("Success! Image data found.")
            return base64.b64encode(image_bytes).decode("utf-8")
        else:
            # إذا لم نجد بيانات، نطبع شكل الجزء للتحليل في الـ Logs
            print(f"Structure Debug: {dir(first_part)}")
            raise Exception("Model returned OK but no image data was found in the response parts.")

    except Exception as e:
        print(f"--- [CRITICAL HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
