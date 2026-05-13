import runpod
import os
import base64
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        # 1. إعداد العميل (v1) لضمان استقرار موديلات 2.5 في منطقتك
        print("--- [STATUS] Worker active. Connecting to Gemini... ---")
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # 2. الترجمة والتحسين
        print("--- [THINKING] Step 1: Optimizing prompt... ---")
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"--- [OK] Prompt Ready: {final_optimized_prompt} ---")

        # 3. طلب التوليد (مع إعدادات الأمان والتوليد الصريح)
        print("--- [THINKING] Step 2: Generating image (Gemini 2.5 Flash)... ---")
        
        # نستخدم ميزة التوليد المباشر لصور Imagen 3 عبر Gemini 2.5
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "SYSTEM: You are a high-end image generation tool. Output ONLY the binary image data. No text.",
                f"Generate a professional photorealistic image: {final_optimized_prompt}"
            ]
        )

        # 4. معالجة الرد ومنع الخطأ الصامت
        print("--- [STATUS] Step 3: Checking response data... ---")
        
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            raise Exception("The model did not return any candidates. Check API limits or safety settings.")

        image_bytes = None
        candidate = response.candidates[0]
        
        # البحث عن البيانات في أجزاء الرد
        if hasattr(candidate.content, 'parts') and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    break
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
                    break

        if image_bytes:
            print("--- [SUCCESS] Image generated successfully! ---")
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{encoded_image}"
        else:
            # طباعة الرد النصي في حال رفض الموديل التوليد لفهم السبب
            debug_text = candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else "Unknown"
            print(f"--- [DEBUG] Model Response: {debug_text} ---")
            raise Exception(f"Model refused to generate image. Reason: {debug_text[:100]}")

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
