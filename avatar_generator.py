import torch
from PIL import Image, ImageOps
import base64
from io import BytesIO
import gc

# [1] الستايلات الاحترافية مدمجة مباشرة لضمان الاستقرار السرعة
AVATAR_STYLES = {
    # 📸 Portrait Mode: تركيز عالي على الوجه مع خلفية مغبشة احترافية
    "photorealistic": "professional cinematic portrait of the person, ultra-detailed eyes, sharp focus on face, shot on 85mm lens, f/1.8, soft bokeh background, blurred backdrop, high-end studio lighting, 8k raw photo, extreme skin detail, subsurface scattering",
    
    # 🎌 Anime Style: تحسين الخطوط والألوان لتصبح كأنمي احترافي
    "anime": "masterpiece, official anime style art of the person, high-quality 2D, studio ghibli aesthetic, cel shaded, clean lineart, vibrant colors, highly detailed expressive eyes, anime character design, best quality",
    
    # 🎮 3D Render: ستايل بيكسار المطور
    "3d_render": "highly detailed 3D Disney Pixar style avatar of the person, stylized digital character, Unreal Engine 5 render, subsurface scattering, cinematic gaming lighting, smooth clay textures, masterfully rendered 3D art",
    
    # 👾 Pixel Art: بيكسل آرت نظيف وحاد
    "pixel_art": "genuine 8-bit pixel art avatar of the person, retro video game sprite, limited color palette, clean square pixels, sharp edges, recognizable facial features in pixel form",
    
    # ✏️ Sketch: رسم يدوي فخم
    "sketch": "raw charcoal sketch of the person on textured paper, messy graphite pencil strokes, hand-drawn artistic lines, rough hatching, elegant minimalist portrait, high contrast black and white",
    
    # 🎨 Abstract: فن تجريدي عصري
    "abstract": "abstract digital art portrait of the person, geometric shapes, double exposure, vibrant neon color splashes, artistic distortion, dreamlike surreal composition, masterpiece"
}

AVATAR_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry, grainy, fuzzy, deformed face, unrecognizable"
)

def generate_avatar(img_pipe, image_b64, prompt, style_key, negative_prompt_input=None):
    try:
        # 2. معالجة الصورة الأصلية (الزوم الذكي خلف الكواليس)
        image_data = base64.b64decode(image_b64)
        init_image = Image.open(BytesIO(image_data)).convert("RGB")
        init_image = ImageOps.exif_transpose(init_image)

        # تحويل لمربع 1024 للتركيز على ملامح الوجه لضمان أعلى دقة
        init_image = ImageOps.fit(init_image, (1024, 1024), method=Image.LANCZOS, centering=(0.5, 0.4))

        # 3. جلب الستايل من القائمة المدمجة أعلاه
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        # 4. هندسة البرومبت النهائي
        final_prompt = f"{prompt}, {style_prompt}, masterpiece, sharp focus on face, detailed skin"
        neg_prompt = negative_prompt_input if negative_prompt_input else AVATAR_NEGATIVE_PROMPT

        # 5. إعدادات القوة (Strength) لضمان ظهور الستايل مع الحفاظ على الشبه
        # 0.60 للستايلات الفنية و 0.45 للواقعية
        custom_strength = 0.60 if style_key != "photorealistic" else 0.45

        # 6. التوليد (Image-to-Image)
        torch.cuda.empty_cache()
        output_image = img_pipe(
            prompt=final_prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            strength=custom_strength,
            num_inference_steps=35,
            guidance_scale=10.0
        ).images[0]

        gc.collect()
        return output_image

    except Exception as e:
        print(f"--- [AVATAR GENERATOR ERROR]: {str(e)} ---")
        try: return init_image
        except: return None
