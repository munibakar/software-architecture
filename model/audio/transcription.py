# Ses dosyalarını metne çeviren fonksiyonu içerir. Whisper modelini kullanarak konuşmaları yazıya döker.


import os
import torch
import logging

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path, job_id):
    try:
        logger.info(f"[{job_id}] Transkripsiyon başlatılıyor: {audio_path}")
        print(f"[{job_id}] Transkripsiyon için ses dosyası: {audio_path}")
        print(f"[{job_id}] Dosya var mı: {os.path.exists(audio_path)}")
        
        # Whisper modelini import et
        try:
            print(f"[{job_id}] Transformers modülünü import ediliyor...")
            from transformers import pipeline
            print(f"[{job_id}] Transformers import edildi")
        except Exception as e:
            print(f"[{job_id}] Transformers import hatası: {str(e)}")
            raise Exception(f"Transformers import hatası: {str(e)}")
        
        # GPU durumunu kontrol et
        try:
            print(f"[{job_id}] GPU kontrolü yapılıyor...")
            print(f"[{job_id}] CUDA kullanılabilir mi: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[{job_id}] Kullanılabilir GPU sayısı: {torch.cuda.device_count()}")
                print(f"[{job_id}] Aktif GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"[{job_id}] GPU kontrolü sırasında hata: {str(e)}")
        
        # Modeli yükle
        print(f"[{job_id}] Whisper modeli yükleniyor...")
        try:
            device = 0 if torch.cuda.is_available() else -1
            print(f"[{job_id}] Cihaz: {device}")
            
            transcriber = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-large-v3",
                chunk_length_s=30,
                device=device
            )
            print(f"[{job_id}] Whisper modeli başarıyla yüklendi")
            
        except Exception as e:
            print(f"[{job_id}] Whisper modeli yükleme hatası: {str(e)}")
            import traceback
            print(f"[{job_id}] Yükleme hata detayları:\n{traceback.format_exc()}")
            raise Exception(f"Whisper modeli yükleme hatası: {str(e)}")
        
        print(f"[{job_id}] Transkripsiyon işlemi başlıyor...")
        try:
            result = transcriber(
                audio_path,
                return_timestamps=True,
                generate_kwargs={"language": "tr"}
            )
            print(f"[{job_id}] Transkripsiyon başarıyla tamamlandı")
            
        except Exception as e:
            print(f"[{job_id}] Transkripsiyon işlemi sırasında hata: {str(e)}")
            import traceback
            print(f"[{job_id}] Transkripsiyon hata detayları:\n{traceback.format_exc()}")
            raise Exception(f"Transkripsiyon işlemi hatası: {str(e)}")
        
        return result["text"], result.get("chunks", [])
        
    except Exception as e:
        print(f"[{job_id}] Transkripsiyon ana fonksiyonunda hata: {str(e)}")
        logger.error(f"[{job_id}] Transkripsiyon hatası: {str(e)}")
        import traceback
        logger.error(f"[{job_id}] Transkripsiyon hata detayları:\n{traceback.format_exc()}")
        return f"Transkripsiyon hatası: {str(e)}", [] 