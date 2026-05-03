# styles_config.py

STYLE_PROMPTS = {
    # 1. الواقعي (مطابق لـ Photorealistic في الصورة)
    "photorealistic": "hyper-realistic portrait, 8k raw photo, masterwork, cinematic lighting, sharp focus, ultra-detailed skin texture, professional photography",
    
    # 2. الأنمي (مطابق لـ Anime Style في الصورة)
    "anime": "high-quality anime style, vibrant colors, clean artistic lines, studio ghibli aesthetic, detailed eyes, masterpiece illustration",
    
    # 3. التجسيم ثلاثي الأبعاد (مطابق لـ 3D Render في الصورة)
    "3d_render": "modern 3d character render, unreal engine 5 style, octane render, smooth textures, subsurface scattering, pixar-like lighting, high-end digital art",
    
    # 4. فن البيكسل (مطابق لـ Pixel Art في الصورة)
    "pixel_art": "high-quality pixel art style, 128-bit aesthetic, retro gaming vibe, clean square blocks, vibrant limited color palette, detailed character sprite",
    
    # 5. الرسم اليدوي (مطابق لـ Sketch في الصورة)
    "sketch": "artistic pencil sketch, charcoal drawing on textured paper, detailed line work, hand-drawn aesthetic, graphite shading, elegant portrait sketch",
    
    # 6. التجريدي (مطابق لـ Abstract في الصورة)
    "abstract": "abstract artistic style, creative geometric patterns, dreamlike colors, double exposure effect, artistic distortion, modern art gallery aesthetic, unique visual composition",

    # الستايلات الاحتياطية (لضمان عمل الكود إذا تم إرسال مسميات قديمة)
    "realistic": "hyper-realistic portrait, 8k raw photo, cinematic lighting, ultra-detailed",
    "pixar": "disney pixar 3d style, cute character, smooth textures, expressive eyes",
    "cartoon": "stylized cartoon illustration, bold outlines, artistic simple colors"
}

# تم تحسين الـ Negative Prompt لمنع ظهور التشوهات في الوجه
NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, messy, lowres, text, watermark, deformed hands, "
    "bad anatomy, extra fingers, poorly drawn face, mutation, deformed eyes, "
    "disfigured, bad art, grainy, low resolution"
)
