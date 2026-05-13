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
        # 1. جلب المفتاح (الذي ينتهي بـ N5UI) من إعدادات RunPod
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # 2. إعداد العميل مع تحديد النسخة v1 (لضمان استقرار موديلات 2.5)
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        mode = job_input.get('mode', 'text')
        style = job_input.get('style', 'photorealistic')
        user_text = job_input.get('prompt', '')

        # 3. تحسين وترجمة النص عبر OpenAI (للحصول على وصف دقيق)
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Optimized Prompt: {final_optimized_prompt}")

        # 4. جلب مقاسات الصورة
        width, height = get_image_dimensions(job_input)

        if mode == 'text':
            # 5. طلب توليد الصورة من الموديل الذي أعطانا استجابة 200 OK
            print("Requesting image generation from gemini-2.5-flash...")
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=f"Generate a cinematic, hyper-realistic 8k photo: {final_optimized_prompt}. Aspect ratio {width}:{height}"
            )
            
            # 6. الفحص الذكي لاستخراج بيانات الصورة (لحل مشكلة NoneType)
            try:
                # الوصول للجزء الأول من الرد
                part = response.candidates[0].content.parts[0]
                
                # فحص وجود البيانات بطرق مختلفة حسب تحديثات API 2026
                image_bytes = None
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
                
                if image_bytes:
                    # تحويل الداتا إلى Base64 وإرسالها
                    return base64.b64encode(image_bytes).decode("utf-8")
                else:
                    # في حال أرجع الموديل نصاً (قد يكون بسبب فلاتر الأمان)
                    error_msg = part.text if hasattr(part, 'text') else "Unknown response format"
                    raise Exception(f"Model did not return image data. Response: {error_msg}")

            except (AttributeError, IndexError) as e:
                print(f"Structure Error: {str(e)}")
                raise Exception("The model responded successfully but the image structure was invalid.")
            
        else:
            # 7. مسار تحويل الصور الشخصية (Avatar Mode)
            image_b64 = job_input.get('image')
            output_img = generate_avatar(image_b64, final_optimized_prompt, style)
            
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        # طباعة الخطأ كاملاً في RunPod Logs لتسهيل تتبعه
        print(f"--- [CRITICAL HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

# ربط الكود بمنصة RunPod Serverless
runpod.serverless.start({"handler": handler})
