import runpod
import os
import base64
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        # استخدام v1 المستقر لضمان التوافق مع حسابك في ألمانيا
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        print("--- [THINKING] Step 1: Translating... ---")
        final_prompt = translate_and_optimize(user_text)

        print(f"--- [THINKING] Step 2: Generating image for: {final_prompt} ---")
        
        # طلب التوليد بدون إعدادات أمان يدوية لتجنب تعارض المسميات
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "GENERATE_IMAGE: No text output. Return only the image blob.",
                f"High-quality professional photo: {final_prompt}"
            ]
        )

        print("--- [STATUS] Step 3: Processing Response... ---")
        
        if not response or not response.candidates:
            return {"error": "No candidates returned. Please try a simpler prompt."}

        image_bytes = None
        candidate = response.candidates[0]
        
        # البحث عن بيانات الصورة في الرد
        if hasattr(candidate.content, 'parts') and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    break
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
                    break

        if image_bytes:
            print("--- [SUCCESS] Image created! ---")
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{encoded_image}"
        else:
            # إذا أرجع الموديل نصاً بدلاً من صورة
            debug_text = candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else "Blocked"
            print(f"--- [REJECTED] Reason: {debug_text} ---")
            return {"error": f"Model refused to generate image. Reason: {debug_text[:50]}"}

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
