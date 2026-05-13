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
        # 1. جلب المفتاح المفعّل (N5UI) من إعدادات RunPod [cite: 10]
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # 2. إعداد العميل (v1) لضمان استقرار موديلات 2.5 [cite: 3]
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # 3. تحسين وترجمة النص عبر OpenAI للحصول على وصف سينمائي [cite: 2]
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Optimized Prompt: {final_optimized_prompt}")

        # 4. جلب مقاسات الصورة
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # 5. طلب توليد الصورة - تم تعديل "التعليمات" لإجبار الموديل على التوليد بدلاً من الوصف 
            print("Requesting image generation from gemini-2.5-flash...")
            
            # نرسل التعليمات كقائمة ليفهم الموديل أننا نريد "فعل" التوليد وليس "نص" الوصف
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=[
                    "Task: Image Generation. Output: Binary Image Data only. Do not return text or descriptions. Create this image now:",
                    f"A cinematic, hyper-realistic 8k photo: {final_optimized_prompt}. Aspect ratio {width}:{height}"
                ]
            )
            
            # 6. فحص الاستجابة واستخراج بيانات الصورة بدقة 
            if not response or not response.candidates:
                raise Exception("Empty response or no candidates from Gemini.")

            image_bytes = None
            # البحث في جميع الأجزاء لضمان التقاط بيانات الصورة [cite: 3]
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    break
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
                    break
                elif hasattr(part, 'blob') and part.blob:
                    image_bytes = part.blob.data
                    break

            if image_bytes:
                print("--- SUCCESS: Image data captured! ---")
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                # إرجاع الصورة بتنسيق Data URI ليتم عرضها فوراً في موقعك
                return f"data:image/png;base64,{encoded_image}"
            else:
                # إذا أرجع الموديل نصاً (كما حدث في السجل السابق)، سنعرف السبب من هنا 
                error_text = response.candidates[0].content.parts[0].text if hasattr(response.candidates[0].content.parts[0], 'text') else "No image data found."
                raise Exception(f"Model returned OK but no image found. Content: {error_text[:200]}...")
            
        else:
            # 7. مسار الأفاتار (تحويل الصور الشخصية)
            image_b64 = job_input.get('image')
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
                
            output_img = generate_avatar(image_b64, final_optimized_prompt, style)
            
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            encoded_avatar = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{encoded_avatar}"

    except Exception as e:
        # تسجيل الخطأ في RunPod Logs 
        print(f"--- [CRITICAL HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
