from google import genai
from PIL import Image
import base64
import io
import os

# الستايلات المدمجة لضمان استجابة فورية
AVATAR_STYLES = {
    "photorealistic": "professional cinematic portrait, 8k raw photo, highly detailed face, sharp focus",
    "anime": "high-quality anime style, studio ghibli aesthetic, vibrant colors, clean lineart",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render, stylized digital character",
    "pixel_art": "genuine 8-bit pixel art, retro video game sprite, clean pixels",
    "sketch": "raw charcoal sketch, artistic lines, elegant minimalist portrait, high contrast",
    "abstract": "abstract digital art, geometric shapes, neon splashes, dreamlike composition"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        # إعداد العميل للمكتبة الجديدة
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        # هندسة الأمر النهائي لـ Gemini
        full_instruction = (
            f"Transform the person in this image into {style_prompt}. "
            f"Context: {prompt}. IMPORTANT: Maintain the person's identity and facial features."
        )

        # تحضير الصورة كجزء من المحتوى
        image_part = {"mime_type": "image/png", "data": image_b64}

        # استخدام generate_content بدلاً من الدالة القديمة 
        response = client.models.generate_content(
            model='gemini-3-flash-image',
            contents=[full_instruction, image_part]
        )

        # تحويل البيانات الثنائية الناتجة إلى صورة PIL
        img_raw = response.candidates[0].content.parts[0].inline_data.data
        return Image.open(io.BytesIO(img_raw))

    except Exception as e:
        print(f"--- [GENERATOR ERROR] ---: {str(e)}")
        # العودة بالصورة الأصلية في حال حدوث خطأ تقني
        return Image.open(io.BytesIO(base64.b64decode(image_b64)))
