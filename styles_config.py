# styles_config.py

STYLE_PROMPTS = {
    # ستايل الأنمي: رسم يدوي ثنائي الأبعاد بالكامل (Cel Shaded)
    "anime": "official anime style art, 2D, flat color, cel shaded, thick outlines, studio ghibli aesthetic, hand-drawn illustration, high-quality vector art, vibrant colors, no realism",
    
    # ستايل الألعاب: تحويل الوجه لشخصية 3D احترافية
    "3d_render": "highly detailed 3D gaming character, Unreal Engine 5 render, Octane render, stylized character design, smooth clay texture, subsurface scattering, cinematic gaming lighting, high-end 3D art",
    
    # ستايل البيكسل: تحويل الملامح لمربعات بيكسل فنية
    "pixel_art": "genuine 8-bit pixel art, retro video game sprite, limited color palette, visible square pixels, nostalgic gaming aesthetic, clean pixel edges, 2D sprite",
    
    # ستايل السكيتش: رسم بالفحم والرصاص على ورق خشن
    "sketch": "raw charcoal sketch on textured paper, messy graphite pencil strokes, hand-drawn artistic lines, rough hatching, elegant minimalist portrait sketch, high contrast black and white",
    
    # ستايل التجريدي: دمج الألوان والأشكال الهندسية بالوجه
    "abstract": "abstract digital art, geometric shapes, double exposure, vibrant neon color splashes, liquid metal textures, dreamlike surreal composition, artistic distortion",
    
    # الستايل الواقعي: بورتريه سينمائي فائق الدقة
    "photorealistic": "professional studio portrait, 8k raw photo, highly detailed skin pores, cinematic rim lighting, sharp focus on eyes, masterpiece, professional photography"
}

# كلمات تمنع الموديل من خلط الستايلات الفنية بالواقعية
NEGATIVE_PROMPT = (
    "photorealistic, real life, 3d, photography, depth of field, blurry, distorted, "
    "lowres, text, watermark, bad anatomy, poorly drawn face, grainy, noisy, messy"
)
