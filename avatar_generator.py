import google.generativeai as genai
from PIL import Image
import base64
from io import BytesIO
import os

# [1] الستايلات الاحترافية مدمجة (تم تحديثها لتناسب ذكاء Gemini)
AVATAR_STYLES = {
    "photorealistic": "professional cinematic portrait, ultra-detailed eyes, sharp focus on face, 8k raw photo, extreme skin detail, subsurface scattering",
    "anime": "official anime style art, studio ghibli aesthetic, cel shaded, clean lineart, vibrant colors, highly detailed eyes",
    "3d_render": "Disney Pixar style 3D avatar, stylized digital character, Unreal Engine 5 render, cinematic lighting, smooth clay textures",
    "pixel_art": "genuine 8-bit pixel art, retro video game sprite, clean square pixels, sharp edges",
    "sketch": "raw charcoal sketch on textured paper, hand-drawn artistic lines, elegant minimalist portrait, high contrast B&W",
    "abstract": "abstract digital art, geometric shapes, vibrant neon color splashes, artistic distortion, dreamlike surreal composition"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        # إعداد المحرك (Nano Banana 2)
        # المفتاح يتم جلبه من إعدادات RunPod التي وضعتها
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-3-flash-image')

        # جلب الستايل المناسب
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        # تحضير الصورة المرفوعة لـ Gemini
        # نرسل الـ Base64 مباشرة كما هو مطلوب في الـ API
        image_part = {
            "mime_type": "image/png", # أو image/jpeg حسب مدخلات موقعك
            "data": image_b64
        }

        # هندسة الأمر النهائي (Instruction)
        # OpenAI قام بترجمة الـ prompt مسبقاً، وهنا نطلب من Gemini تطبيق الستايل
        full_instruction = (
            f"Transform the person in this image into this style: {style_prompt}. "
            f"Context: {prompt}. "
            f"IMPORTANT: Maintain the person's original facial features, identity, and gender. "
            f"The output must be a high-quality stylized portrait."
        )

        # إرسال الطلب لـ Google
        response = model.generate_content([full_instruction, image_part])

        # تحويل النتيجة (Base64) القادمة من Google إلى PIL Image ليعالجها الـ Handler
        generated_image_data = base64.b64decode(response.candidates[0].content.parts[0].inline_data.data)
        output_image = Image.open(BytesIO(generated_image_data))

        return output_image

    except Exception as e:
        print(f"--- [AVATAR GENERATOR ERROR]: {str(e)} ---")
        # في حال الخطأ، نعود بالصورة الأصلية لكي لا يفصل الموقع
        try:
            image_data = base64.b64decode(image_b64)
            return Image.open(BytesIO(image_data))
        except:
            return None
