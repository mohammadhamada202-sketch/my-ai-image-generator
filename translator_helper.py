# translator_helper.py
import os
from openai import OpenAI

# قراءة المفتاح من إعدادات RunPod (Environment Variables)
# تأكد أنك سميت المتغير في RunPod باسم OPENAI_API_KEY
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def translate_and_optimize(user_input):
    # إذا لم يكن هناك مدخلات، نعود فوراً
    if not user_input:
        return user_input
        
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional prompt engineer. Translate the user's input to English and enhance it for AI image generation. Return only the optimized English prompt."
                },
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        # طباعة الخطأ في الـ Logs لنعرف السبب (مثل نقص الرصيد أو مفتاح خاطئ)
        print(f"OpenAI Error: {str(e)}")
        # في حال الفشل، نعود بالنص الأصلي
        return user_input
