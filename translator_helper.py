import os
from openai import OpenAI

def translate_and_optimize(user_input):
    # التأكد من وجود نص للإدخال
    if not user_input or user_input.strip() == "":
        return user_input

    # قراءة المفتاح من إعدادات RunPod
    api_key = os.getenv("OPENAI_API_KEY")
    
    # التحقق من وجود المفتاح لتجنب توقف السيرفر
    if not api_key:
        print("CRITICAL: OPENAI_API_KEY is not set in RunPod environment variables.")
        return user_input

    client = OpenAI(api_key=api_key)

    try:
        # إرسال الطلب لـ OpenAI (gpt-3.5-turbo)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional prompt engineer. Translate the input to English and enhance it with artistic details for high-quality AI image generation. Return ONLY the final English prompt."
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        optimized_text = response.choices[0].message.content
        print(f"Success! Translated Prompt: {optimized_text}")
        return optimized_text

    except Exception as e:
        # طباعة الخطأ في Logs لتعرف السبب (رصيد، مفتاح، إلخ)
        print(f"OpenAI Error: {str(e)}")
        return user_input
