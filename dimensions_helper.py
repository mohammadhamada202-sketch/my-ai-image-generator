# dimensions_helper.py

def get_dimensions(job_input):
    """
    تحديد المقاسات بناءً على الـ aspect_ratio المرسل من الموقع
    أو استخدام العرض والطول المباشر إذا توفرا.
    """
    # 1. جلب اسم المقاس المرسل من الموقع (مثل 9:16 أو 1:1)
    aspect_ratio = job_input.get('aspect_ratio', '1:1')

    # 2. تحديد الأبعاد الافتراضية بناءً على المسمى
    if aspect_ratio == "9:16":
        # المقاس الطولي (Portrait)
        width, height = 768, 1344
    elif aspect_ratio == "16:9":
        # المقاس العرضي (Landscape)
        width, height = 1344, 768
    elif aspect_ratio == "1:1":
        # المقاس المربع (Square)
        width, height = 1024, 1024
    else:
        # إذا لم يُرسل مسمى، نبحث عن أرقام مباشرة width/height أو نستخدم المربع كافتراضي
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

    # 3. التأكد من أن المقاسات مضاعفات العدد 8 (مطلوب لـ SDXL لضمان الجودة)
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    print(f"DEBUG: Selected Dimensions -> {width}x{height} for Ratio -> {aspect_ratio}")
    return width, height
