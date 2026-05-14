import runpod
import os
import uuid  # لتوليد أسماء فريدة للصور
from google import genai
from supabase import create_client # مكتبة الربط مع سوبابيس
from translator_helper import translate_and_optimize

# إعداد بيانات الربط مع Supabase (يفضل وضعها في Environment Variables في RunPod)
SUPABASE_URL = os.environ.get("SUPABASE_URL") # من Project Settings -> API
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # أو الـ anon key
BUCKET_NAME = "MyFirstImagesTest1"

# إنشاء العميل
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        print("--- [STATUS] Step 1: Optimizing prompt... ---")
        final_prompt = translate_and_optimize(user_text)

        print(f"--- [STATUS] Step 2: Generating image via Gemini 2.5 ---")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "TASK: GENERATE_IMAGE. NO TEXT OUTPUT. RETURN ONLY THE IMAGE DATA.",
                f"Professional high-quality 4K photo: {final_prompt}"
            ]
        )

        if response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content.parts:
                image_bytes = None
                
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_bytes = part.inline_data.data
                        break
                    elif hasattr(part, 'data') and part.data:
                        image_bytes = part.data
                        break
                
                if image_bytes:
                    print("--- [STATUS] Step 3: Uploading to Supabase... ---")
                    
                    # 1. إنشاء اسم فريد للصورة لمنع التكرار
                    file_name = f"gen_{uuid.uuid4()}.png"
                    
                    # 2. عملية الرفع لـ Supabase
                    storage = supabase.storage.from_(BUCKET_NAME)
                    storage.upload(
                        path=file_name,
                        file=image_bytes,
                        file_options={"content-type": "image/png"}
                    )
                    
                    # 3. استخراج الرابط العام (Public URL)
                    image_url = storage.get_public_url(file_name)
                    
                    print(f"--- [SUCCESS] Image live at: {image_url} ---")
                    
                    # نرجع الرابط فقط، المتصفح سيفرح بهذا الرد الصغير!
                    return {"image_url": image_url}
                
                else:
                    return {"error": "Model sent text instead of image data."}
        
        return {"error": "Failed to extract image from Gemini response."}

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
