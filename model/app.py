# Ana uygulama dosyası. Flask web sunucusunu başlatır, CORS ayarlarını yapar, loglama sistemini kurar ve API rotalarını kaydeder.

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from dotenv import load_dotenv

# Modülleri import et
from .api.routes import register_routes
from .jobs.processor import results_cache

# Uygulama oluşturma
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Her istekten sonra CORS başlıklarını ekleyelim
@app.after_request
def after_request(response):
    print("İstek alındı:", request.method, request.path)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Loglama konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Çevre değişkenlerini yükle
load_dotenv()

# FFmpeg'i otomatik olarak yapılandır
try:
    backend_ffmpeg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    ffmpeg_exe_path = os.path.join(backend_ffmpeg_path, "ffmpeg.exe")
    
    if os.path.exists(ffmpeg_exe_path):
        print(f"FFmpeg bulundu: {ffmpeg_exe_path}")
        # FFmpeg'i PATH'e ekle
        os.environ["PATH"] = f"{backend_ffmpeg_path};{os.environ['PATH']}"
        print(f"FFmpeg PATH'e eklendi: {backend_ffmpeg_path}")
    else:
        print(f"FFmpeg bulunamadı: {ffmpeg_exe_path}")
except Exception as e:
    print(f"FFmpeg yapılandırma hatası: {str(e)}")

# API rotalarını kaydet
register_routes(app)

# Ana fonksiyon
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 