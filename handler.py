import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        # 1. إعداد المفتاح والعميل (N5UI) مع تحديد الإصدار v1 المستقر لعام 2026
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # 2. الترجمة والتحسين لضمان فهم Gemini العميق للمشهد
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Executing for Prompt: {final_optimized_prompt}")

        # 3. طلب التوليد - تم تغيير الموديل ليكونImagen 3 صراحةً إذا كان متاحاً
        # أو إجبار Gemini 2.5 Flash على وضع التوليد فقط
        print("Requesting image from gemini-2.5-flash...")
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[
                "TASK: GENERATE_IMAGE. NO TEXT OUTPUT. RETURN ONLY THE INLINE_DATA BLOB.",
                f"A professional 8k cinematic photo: {final_optimized_prompt}"
            ]
        )

        # 4. البحث المعمق عن البيانات الثنائية في كل أجزاء الاستجابة
        if not response or not response.candidates:
            raise Exception("No response from Gemini API.")

        image_bytes = None
        for part in response.candidates[0].content.parts:
            # التحقق من وجود بيانات الصورة (inline_data) التي يحتاجها الموقع
            if hasattr(part, 'inline_data') and part.inline_data:
                image_bytes = part.inline_data.data
                break
            elif hasattr(part, 'data') and part.data:
                image_bytes = part.data
                break

        if image_bytes:
            print("--- SUCCESS: Image generated and captured! ---")
            # تحويل البيانات لـ Base64 مع الترويسة لضمان عرضها فوراً في الموقع
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{encoded_image}"
        else:
            # استخراج النص التوضيحي في حال فشل التوليد لفهم السبب (مثل فلاتر الأمان)
            debug_text = response.candidates[0].content.parts[0].text if hasattr(response.candidates[0].content.parts[0], 'text') else "No image found"
            raise Exception(f"Model returned OK but no image data found. Content preview: {debug_text[:50]}")

    except Exception as e:
        print(f"--- [CRITICAL HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل العامل (Worker) على RunPod
runpod.serverless.start({"handler": handler})
