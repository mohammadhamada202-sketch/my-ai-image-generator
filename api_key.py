# api_key.py
import os
import base64
import subprocess
import sys

# --- 🛠️ تثبيت مكتبة جوجل تلقائياً في الخلفية لو نقصت في RunPods ---
try:
    from google.cloud import aiplatform
except ImportError:
    print("--- [SYSTEM] google-cloud-aiplatform not found. Installing now... ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-aiplatform"])
    print("--- [SYSTEM] google-cloud-aiplatform installed successfully! ---")

def initialize_google_vertex_env():
    try:
        current_dir = os.path.dirname(__file__)
        encrypted_file_path = os.path.join(current_dir, "secure_matrix.txt")
        
        if not os.path.exists(encrypted_file_path):
            print("--- [VERTEX CONFIG ERROR] Secure matrix file not found! ---")
            return

        with open(encrypted_file_path, "r", encoding="utf-8") as f:
            encrypted_content = f.read().strip()
        
        # فك التشفير في الذاكرة المؤقتة لـ RunPods
        decoded_bytes = base64.b64decode(encrypted_content)
        json_string = decoded_bytes.decode("utf-8")
        
        google_key_path = "/tmp/google_vertex_key.json"
        with open(google_key_path, "w", encoding="utf-8") as json_file:
            json_file.write(json_string)
            
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_key_path
        
        if "GOOGLE_PROJECT_ID" not in os.environ:
            os.environ["GOOGLE_PROJECT_ID"] = "hip-gecko-496121-f9"
        if "GOOGLE_LOCATION" not in os.environ:
            os.environ["GOOGLE_LOCATION"] = "us-central1"
            
        print("--- [SECURITY] Google Cloud Vertex Environment Initialized Successfully ---")
        
    except Exception as e:
        print(f"--- [SECURITY ERROR] Failed to initialize Vertex env: {str(e)} ---")

# تشغيل الفك التلقائي بمجرد الاستدعاء
initialize_google_vertex_env()
