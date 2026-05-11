from google import genai
from PIL import Image
import base64
import io
import os

AVATAR_STYLES = {
    "photorealistic": "professional cinematic portrait, 8k raw photo, highly detailed face",
    "anime": "high-quality anime style, studio ghibli aesthetic, clean lineart",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render",
    "pixel_art": "genuine 8-bit pixel art, retro video game sprite",
    "sketch": "raw charcoal sketch, artistic lines, high contrast",
    "abstract": "abstract digital art, neon splashes, artistic distortion"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        # إعداد العميل
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        # صياغة التعليمات
        full_instruction = (
            f"Transform the person in this image into {style_prompt}. "
            f"Context: {prompt}. Maintain identity and facial features."
        )

        # تحضير الجزء الخاص بالصورة
        image_part = {"mime_type": "image/png", "data": image_b64}

        # طلب التوليد
        response = client.models.generate_content(
            model='gemini-3-flash-image',
            contents=[full_instruction, image_part]
        )

        # استخراج الصورة
        img_raw = response.candidates[0].content.parts[0].inline_data.data
        return Image.open(io.BytesIO(img_raw))

    except Exception as e:
        print(f"--- [GENERATOR ERROR] ---: {str(e)}")
        # في حال الفشل نرجع الصورة الأصلية للمستخدم
        return Image.open(io.BytesIO(base64.b64decode(image_b64)))
