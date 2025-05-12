# Ana uygulama dosyası. Flask web sunucusunu başlatır, CORS ayarlarını yapar, loglama sistemini kurar ve API rotalarını kaydeder.

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import torch
from dotenv import load_dotenv

# Modülleri import et
from .api.routes import register_routes
from .jobs.processor import results_cache

# GPU kullanımını optimize etmek için ayarlar
def configure_gpu():
    if torch.cuda.is_available():
        # GPU kullanımını optimize et
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Bellek kullanımını optimize et
        torch.cuda.empty_cache()
        
        # Veri tipi tutarlılığı için float32 kullan
        torch.set_default_dtype(torch.float32)
        
        logger = logging.getLogger(__name__)
        logger.info(f"GPU yapılandırıldı: {torch.cuda.get_device_name(0)}")
        print(f"GPU yapılandırıldı: {torch.cuda.get_device_name(0)}")
        return True
    return False

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

# GPU yapılandırmasını çalıştır
has_gpu = configure_gpu()
if has_gpu:
    logger.info("GPU kullanımı etkinleştirildi")
else:
    logger.info("GPU bulunamadı, CPU kullanılacak")

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