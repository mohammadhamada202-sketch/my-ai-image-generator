import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        # المفاتيح الاحترافية تتطلب v1 لضمان استقرار المشاريع المدفوعة
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')
        final_optimized_prompt = translate_and_optimize(user_text)

        # مصفوفة الموديلات الوحيدة الممكنة لحسابك حالياً
        # بما أنك فعلت Gemini API، فالموديل gemini-1.5-flash هو الأكثر استقراراً
        test_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'imagen-3.0-generate-002']
        
        response = None
        for model_id in test_models:
            try:
                print(f"Testing access to: {model_id}...")
                response = client.models.generate_content(
                    model=model_id, 
                    contents=f"Generate a cinematic, hyper-realistic 8k image: {final_optimized_prompt}"
                )
                if response:
                    print(f"SUCCESS! Connection established with: {model_id}")
                    break
            except Exception as e:
                print(f"Failed {model_id}: {str(e)}")
                continue

        if not response:
            raise Exception("All enabled models returned 404. Check if Gemini API is fully propagated.")

        # استخراج الصورة
        image_bytes = response.candidates[0].content.parts[0].inline_data.data
        return base64.b64encode(image_bytes).decode("utf-8")

    except Exception as e:
        print(f"--- [HANDLER ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
