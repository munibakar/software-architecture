# API rotalarını tanımlar. İstemcilerin erişebileceği HTTP endpoint'lerini içerir.

from flask import request, jsonify
import os
import time
from threading import Thread
import logging

from ..jobs.processor import process_job, results_cache

logger = logging.getLogger(__name__)

def register_routes(app):
    @app.route('/api/process', methods=['POST'])
    def start_processing():
        try:
            print("POST /api/process endpoint'i çağrıldı")
            data = request.json
            print(f"Alınan veri: {data}")
            
            audio_path = data.get('audio_path')
            print(f"Ses dosyası yolu: {audio_path}")
            
            if not audio_path:
                print("HATA: Ses dosyası yolu belirtilmedi")
                return jsonify({"error": "Audio path is required"}), 400
            
            # Dosyanın varlığını ve erişilebilirliğini kontrol et
            print(f"Dosya var mı: {os.path.exists(audio_path)}")
            if os.path.exists(audio_path):
                print(f"Dosya boyutu: {os.path.getsize(audio_path)} bytes")
                print(f"Dosya okunabilir mi: {os.access(audio_path, os.R_OK)}")
            
            if not os.path.exists(audio_path):
                print(f"HATA: Ses dosyası bulunamadı: {audio_path}")
                return jsonify({"error": f"Audio file not found: {audio_path}"}), 404
            
            job_id = str(int(time.time()))
            print(f"Oluşturulan job_id: {job_id}")
            
            # İşlemi arka planda başlat
            print("Arka plan işlemi başlatılıyor...")
            thread = Thread(target=process_job, args=(audio_path, job_id))
            thread.daemon = True
            thread.start()
            
            print(f"İşlem başlatıldı, job_id: {job_id}")
            return jsonify({
                "message": "Processing started",
                "job_id": job_id
            })
            
        except Exception as e:
            print(f"API hatası: {str(e)}")
            logger.error(f"API hatası: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/status/<job_id>', methods=['GET'])
    def get_job_status(job_id):
        print(f"GET /api/status/{job_id} endpoint'i çağrıldı")
        
        if job_id not in results_cache:
            print(f"HATA: Job ID bulunamadı: {job_id}")
            return jsonify({"status": "not_found"}), 404
        
        print(f"Durum yanıtı: {results_cache[job_id]}")
        return jsonify(results_cache[job_id])

    @app.route('/api/result/<job_id>', methods=['GET'])
    def get_job_result(job_id):
        print(f"GET /api/result/{job_id} endpoint'i çağrıldı")
        
        if job_id not in results_cache:
            print(f"HATA: Job ID bulunamadı: {job_id}")
            return jsonify({"error": "Job not found"}), 404
        
        job_result = results_cache[job_id]
        
        if job_result["status"] != "completed":
            print(f"HATA: İşlem henüz tamamlanmadı. Durum: {job_result['status']}")
            return jsonify({
                "status": job_result["status"],
                "error": job_result.get("error", "Job is still processing")
            }), 400
        
        print(f"Sonuç başarıyla döndürüldü")
        return jsonify(job_result)

    # İlave test endpoint'i
    @app.route('/api/test', methods=['GET', 'POST'])
    def test_api():
        print(f"Test API çağrısı: {request.method}")
        if request.method == 'POST':
            print(f"POST veri: {request.json}")
        
        return jsonify({
            "status": "success",
            "message": "Model servisi çalışıyor!",
            "time": time.time()
        }) 