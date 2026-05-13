import runpod
import os
import base64
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        # إعداد المفتاح والعميل باستخدام النسخة v1 المستقرة
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        print("--- [STATUS] Step 1: Optimizing prompt... ---")
        final_prompt = translate_and_optimize(user_text)

        print(f"--- [STATUS] Step 2: Generating image for: {final_prompt} ---")
        
        # تم حذف إعدادات الأمان يدوياً لتجنب الخطأ 400 INVALID_ARGUMENT
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "TASK: GENERATE_IMAGE. NO TEXT OUTPUT. RETURN ONLY THE IMAGE DATA.",
                f"Professional high-quality photo: {final_prompt}"
            ]
        )

        print("--- [STATUS] Step 3: Extracting Image... ---")
        
        if not response or not response.candidates:
            raise Exception("No candidates returned from Gemini.")

        image_bytes = None
        candidate = response.candidates[0]
        
        # استخراج بيانات الصورة (inline_data) بشكل آمن لضمان نجاح العرض
        if hasattr(candidate.content, 'parts') and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    break
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
                    break

        if image_bytes:
            print("--- [SUCCESS] Image created successfully! ---")
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            # إضافة ترويسة Base64 لضمان ظهور الصورة فوراً على موقعك
            return f"data:image/png;base64,{encoded_image}"
        else:
            # في حال أرجع الموديل نصاً بدلاً من صورة لفهم السبب
            debug_text = candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else "Blocked"
            print(f"--- [DEBUG] Rejection Reason: {debug_text} ---")
            raise Exception(f"Model refused. Reason: {debug_text[:100]}")

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
