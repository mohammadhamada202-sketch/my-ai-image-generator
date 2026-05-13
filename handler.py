import runpod
import os
import base64
from google import genai
from google.genai import types
from translator_helper import translate_and_optimize

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        # استخدام نسخة v1 المستقرة في ألمانيا
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # الخطوة 1: الترجمة
        print("--- [THINKING] Step 1: Optimizing prompt... ---")
        final_optimized_prompt = translate_and_optimize(user_text)

        # الخطوة 2: التوليد (مع إضافة فلاتر الأمان لتقليل وقت الفحص)
        print("--- [THINKING] Step 2: Generating image (Fast Track)... ---")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "GENERATE_IMAGE: No text description. Output binary data only.",
                f"Cinematic realistic photo: {final_optimized_prompt}"
            ],
            # تقليل قيود الأمان لسرعة الاستجابة ومنع التعليق
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(category='HATE_SPEECH', threshold='BLOCK_NONE'),
                    types.SafetySetting(category='HARASSMENT', threshold='BLOCK_NONE'),
                ]
            )
        )

        # الخطوة 3: استخراج البيانات
        print("--- [STATUS] Step 3: Checking data... ---")
        if not response or not response.candidates:
            raise Exception("No response from Gemini.")

        image_bytes = None
        candidate = response.candidates[0]
        if hasattr(candidate.content, 'parts') and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    break
        
        if image_bytes:
            print("--- [SUCCESS] Image generated! ---")
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{encoded_image}"
        else:
            # إذا أرجع الموديل نصاً بدلاً من صورة، سنعرف السبب فوراً
            reason = candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else "Safety block"
            print(f"--- [DEBUG] Reason: {reason} ---")
            raise Exception(f"Model refused. Reason: {reason[:50]}")

    except Exception as e:
        print(f"--- [ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
