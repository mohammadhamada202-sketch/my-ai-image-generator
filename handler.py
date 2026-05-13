import runpod
import os
import base64
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        # إعداد العميل بنسخة v1 المستقرة
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        print("--- [STATUS] Step 1: Optimizing prompt... ---")
        final_prompt = translate_and_optimize(user_text)

        print(f"--- [STATUS] Step 2: Generating image for: {final_prompt} ---")
        
        # طلب التوليد المباشر
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "TASK: GENERATE_IMAGE. NO TEXT OUTPUT. RETURN ONLY THE IMAGE DATA.",
                f"Professional high-quality photo: {final_prompt}"
            ]
        )

        print("--- [STATUS] Step 3: Extracting Image (Safe Mode)... ---")
        
        # التحقق الآمن من وجود الرد والمرشحين (Candidates)
        if response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # التحقق من وجود المحتوى والأجزاء
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                image_bytes = None
                
                # البحث في الأجزاء عن البيانات الثنائية
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
                    # فحص إذا كان هناك رد نصي بدلاً من الصورة
                    text_reply = candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else "No image data"
                    print(f"--- [REJECTED] Model sent text instead of image: {text_reply} ---")
                    return {"error": f"Model refused to generate image. Reason: {text_reply[:100]}"}
            else:
                return {"error": "Response candidate has no content parts."}
        else:
            return {"error": "No candidates found in Gemini response."}

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
