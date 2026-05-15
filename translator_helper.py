import os
from openai import OpenAI

def get_epic_prompt(user_input):
    """
    يحول أي فكرة بسيطة إلى مشهد سينمائي خارق.
    """
    if not user_input or user_input.strip() == "":
        return user_input

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # الموديل الأقوى للتخيل
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a visionary Concept Artist and Cinematographer. "
                        "Your goal is to transform the user's idea into a 'breathtaking masterpiece'. "
                        "1. Translate to English. "
                        "2. Add epic details: cinematic lighting (volumetric fog, rim lighting), "
                        "dramatic environment (floating islands, cyberpunk neon, ancient ruins), "
                        "and high-end textures (8k, unreal engine 5 render, ray-tracing). "
                        "3. Make it 'Epic' and 'Legendary'. "
                        "Return ONLY the final English prompt."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.9 # درجة إبداع عالية جداً
        )
        epic_prompt = response.choices[0].message.content
        print(f"--- [EPIC PROMPT GENERATED] ---: {epic_prompt}")
        return epic_prompt
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return user_input
