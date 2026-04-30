# dimensions_helper.py

def get_dimensions(job_input):
    # المقاسات الافتراضية
    width = job_input.get('width', 1024)
    height = job_input.get('height', 1024)
    
    # التأكد من أن المقاسات مضاعفات العدد 8 (مطلوب لـ SDXL)
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    return width, height