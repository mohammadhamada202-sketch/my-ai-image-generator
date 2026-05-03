# styles_config.py

STYLE_PROMPTS = {
    # أضفنا ستايل باسم realistic لحل المشكلة التي ظهرت في الـ Logs
    "realistic": "hyper-realistic portrait, 8k raw photo, cinematic lighting, ultra-detailed skin texture",
    "photorealistic": "professional studio portrait, 8k raw photo, cinematic lighting, masterpiece",
    "anime": "official anime style art, 2D, flat color, cel shaded, vibrant colors, hand-drawn illustration",
    "3d_render": "highly detailed 3D gaming character, Unreal Engine 5 render, stylized character design",
    "pixel_art": "genuine 8-bit pixel art, retro video game sprite, visible square pixels",
    "sketch": "raw charcoal sketch on textured paper, hand-drawn artistic lines, rough hatching",
    "abstract": "abstract digital art, geometric shapes, vibrant neon color splashes"
}

NEGATIVE_PROMPT = (
    "photorealistic, real life, 3d, photography, depth of field, blurry, distorted, "
    "lowres, text, watermark, bad anatomy, poorly drawn face, grainy, noisy, messy"
)
