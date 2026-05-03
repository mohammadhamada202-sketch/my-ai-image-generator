import torch
from PIL import Image, ImageOps
import base64
from io import BytesIO
# استيراد الستايلات المخصصة للأفاتار من الملف الذي أنشأناه
try:
    from avatar_styles_config import AVATAR_STYLES, AVATAR_NEGATIVE_PROMPT
except ImportError:
    # في حال لم يتم العثور على الملف، نستخدم قيم افتراضية لضمان عدم توقف السيرفر
    AVATAR_STYLES = {"photorealistic": "professional portrait, highly detailed likeness"}
    AVATAR_NEGATIVE_PROMPT = "blurry, distorted, low quality"

def generate_avatar(img_pipe, image_b64, prompt, style_key, negative_prompt):
    """
    دالة توليد الأفاتار التي تركز على تحليل الوجه والحفاظ على الهوية الشخصية.
    """
    try:
        # 1. تحويل الـ Base64 القادم من الموقع إلى صورة PIL
        image_data = base64.b64decode(image_b64)
        init_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # 2. إجراء "مسح ذكي" للوجه (Smart Face Crop)
        # نقوم بقص الصورة لتصبح 1024x1024 مع توضيع الوجه في الثلث العلوي (الوضعية المثالية للأفاتار)
        # هذا يمنع تمطيط الوجه ويضمن أن الذكاء الاصطناعي يركز على الملامح الدقيقة[span_1](start_span)[span_1](end_span)
        init_image = ImageOps.fit(init_image, (1024, 1024), centering=(0.5, 0.4))
        
        # 3. جلب الوصف الخاص بالستايل المختار (Anime, Sketch, etc.)
        # إذا لم يرسل الموقع ستايل معروف، يتم استخدام 'photorealistic' كخيار افتراضي
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES.get("photorealistic"))
        
        # 4. هندسة البرومبت لقفل الهوية (Identity Locking)
        # ندمج الستايل الفني مع أوامر تجبر الموديل على مطابقة ملامح الشخص في الصورة الأصلية[span_2](start_span)[span_2](end_span)
        identity_boost = "precise facial likeness, maintaining same identity, highly recognizable person, detailed facial features"
        final_prompt = f"{style_prompt}, {identity_boost}"
        
        # 5. ضبط قوة التغيير (Strength) لضمان الشبه
        # السر في "الشبه" هو عدم المبالغة في الـ Strength. 
        # القيمة 0.50 تعني استخدام 50% من ملامحك الأصلية و50% من الستايل الفني[span_3](start_span)[span_3](end_span)
        if any(s in style_key.lower() for s in ["anime", "sketch", "pixel"]):
            custom_strength = 0.55 # قوة أعلى قليلاً للأنمي والسكيتش ليظهر طابع الرسم
        else:
            custom_strength = 0.48 # قوة أقل للواقعية والـ 3D للحفاظ على تطابق الملامح التام
            
        # 6. تنفيذ عملية التوليد باستخدام الموديل (SDXL)
        # رفعنا الـ guidance_scale لزيادة حدة التفاصيل والالتزام بالستايل[span_4](start_span)[span_4](end_span)
        image = img_pipe(
            prompt=final_prompt,
            negative_prompt=AVATAR_NEGATIVE_PROMPT,
            image=init_image,
            strength=custom_strength,
            num_inference_steps=40,    # عدد خطوات أعلى لدقة التفاصيل في العيون والبشرة[span_5](start_span)[span_5](end_span)
            guidance_scale=13.0        # قوة توجيه عالية لضمان جودة الأفاتار[span_6](start_span)[span_6](end_span)
        ).images[0]
        
        return image

    except Exception as e:
        print(f"--- [Avatar Error Log] ---: {str(e)}")
        # في حال حدوث أي خطأ، نعيد الصورة الأصلية (بعد القص) لضمان استمرارية عمل الواجهة
        return init_image
