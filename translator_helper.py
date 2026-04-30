# translator_helper.py
from openai import OpenAI

# ضع مفتاح OpenAI الخاص بك هنا
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def translate_and_optimize(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional prompt engineer. Translate the user's input to English and enhance it for AI image generation. Return only the optimized English prompt."},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception:
        # في حال حدوث خطأ في الاتصال، نستخدم النص الأصلي
        return user_input