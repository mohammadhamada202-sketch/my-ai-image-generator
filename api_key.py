# api_key.py
import os
import base64

def initialize_google_vertex_env():
    """
    دالة تقوم بقراءة الملف المشفر وفك تشفيره في الذاكرة لتجهيز سيرفر RunPods
    """
    try:
        # 1. قراءة الملف المشفر الذي رفعناه على جيت هاب
        current_dir = os.path.dirname(__file__)
        encrypted_file_path = os.path.join(current_dir, "secure_matrix.txt")
        
        if not os.path.exists(encrypted_file_path):
            print("--- [VERTEX CONFIG ERROR] Secure matrix file not found! ---")
            return

        with open(encrypted_file_path, "r", encoding="utf-8") as f:
            encrypted_content = f.read().strip()
        
        # 2. فك تشفير الـ Base64 في الذاكرة فوراً
        decoded_bytes = base64.b64decode(encrypted_content)
        json_string = decoded_bytes.decode("utf-8")
        
        # 3. كتابة الملف داخل المجلد المؤقت الآمن والمعزول بسيرفر RunPods (/tmp)
        google_key_path = "/tmp/google_vertex_key.json"
        with open(google_key_path, "w", encoding="utf-8") as json_file:
            json_file.write(json_string)
            
        # 4. حقن مسار المفتاح في متغيرات البيئة ليتعرف نظام جوجل عليه تلقائياً
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_key_path
        
        # 5. تعيين بقية متغيرات جوجل الأساسية احتياطياً لو لم تكن بالـ RunPods UI
        if "GOOGLE_PROJECT_ID" not in os.environ:
            os.environ["GOOGLE_PROJECT_ID"] = "hip-gecko-496121-f9"
        if "GOOGLE_LOCATION" not in os.environ:
            os.environ["GOOGLE_LOCATION"] = "us-central1"
            
        print("--- [SECURITY] Google Cloud Vertex Environment Initialized Successfully ---")
        
    except Exception as e:
        print(f"--- [SECURITY ERROR] Failed to initialize Vertex env: {str(e)} ---")

# تشغيل الدالة فوراً بمجرد استدعاء الملف في الـ Handler
initialize_google_vertex_env()
