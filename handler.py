import os
import uuid
import sys
import runpod
import time

# قمنا بوضع الاستيراد داخل الدالة لضمان صيد أي خطأ في المكتبات
def handler(job):
    print("--- [START] HANDLER V6.0 - CHECKING WHY NO CREDIT IS USED ---")
    try:
        # 1. فحص المكتبات (هل هي مثبتة أصلاً؟)
        try:
            from google import genai
            from supabase import create_client
            print("--- [DEBUG] Libraries loaded successfully ---")
        except Exception as lib_err:
            print(f"--- [CRITICAL] Library Import Failed: {str(lib_err)} ---")
            return {"error": f"Import error: {str(lib_err)}"}

        # 2. فحص المفاتيح
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("--- [CRITICAL] GEMINI_API_KEY IS EMPTY! ---")
            return {"error": "API Key Missing"}

        # 3. محاولة الاتصال بـ Gemini (هنا تبدأ التكلفة)
        print("--- [STEP] Initializing Gemini Client... ---")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job.get('input', {})
        prompt = job_input.get('prompt', 'Apple')

        print(f"--- [STEP] Requesting Image for prompt: {prompt} ---")
        
        # محاولة التوليد مع صيد الخطأ الخاص بجوجل
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[f"High-quality 4K image of {prompt}"]
            )
            print("--- [DEBUG] Gemini API call finished ---")
        except Exception as gemini_err:
            print(f"--- [FAILED] Gemini API Refused the call: {str(gemini_err)} ---")
            return {"error": f"Gemini Error: {str(gemini_err)}"}

        # 4. فحص هل وصلت بيانات فعلاً؟
        image_bytes = None
        if response and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data'): image_bytes = part.inline_data.data
                elif hasattr(part, 'data'): image_bytes = part.data
        
        if not image_bytes:
            print("--- [WARNING] Gemini responded OK but sent NO IMAGE DATA ---")
            return {"error": "No image data returned"}

        print(f"--- [SUCCESS] Image Data Received! Size: {len(image_bytes)} bytes ---")
        
        # 5. الرفع السريع لـ Supabase
        sb_url = os.environ.get("SUPABASE_URL", "").strip()
        sb_key = os.environ.get("SUPABASE_KEY", "").strip()
        sb = create_client(sb_url, sb_key)
        
        file_name = f"final_test_{uuid.uuid4().hex[:5]}.png"
        sb.storage.from_("MyFirstImagesTest1").upload(file_name, image_bytes)
        
        url = sb.storage.from_("MyFirstImagesTest1").get_public_url(file_name)
        print(f"--- [DONE] Image Uploaded to: {url} ---")
        
        return {"url": url, "status": "success"}

    except Exception as fatal_e:
        print(f"--- [FATAL] Unexpected Crash: {str(fatal_e)} ---")
        return {"error": str(fatal_e)}

runpod.serverless.start({"handler": handler})
