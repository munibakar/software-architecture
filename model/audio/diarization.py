# Konuşmacı ayrıştırma fonksiyonlarını içerir. Farklı konuşmacıları birbirinden ayırır.

import os
import torch
import logging

logger = logging.getLogger(__name__)

def diarize_audio(audio_path, job_id):
    try:
        logger.info(f"[{job_id}] Konuşmacı ayrıştırma başlatılıyor: {audio_path}")
        print(f"[{job_id}] Konuşmacı ayrıştırma için ses dosyası: {audio_path}")
        
        # Environment variable kontrolü yap
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print(f"[{job_id}] UYARI: HUGGINGFACE_TOKEN çevre değişkeni bulunamadı!")
            print(f"[{job_id}] .env dosyasını kontrol edin ve gerekli token'ı ayarlayın")
            # UYARI: Mockup veriler kullanmak yerine hata döndür
            raise Exception("HUGGINGFACE_TOKEN çevre değişkeni bulunamadı. Konuşmacı ayrıştırma yapılamaz.")
        
        # Pyannote.audio modelini import et
        try:
            print(f"[{job_id}] Pyannote.audio modülünü import ediliyor...")
            from pyannote.audio import Pipeline
            print(f"[{job_id}] Pyannote.audio import edildi")
        except Exception as e:
            print(f"[{job_id}] Pyannote.audio import hatası: {str(e)}")
            import traceback
            print(f"[{job_id}] Import hata detayları:\n{traceback.format_exc()}")
            raise Exception(f"Pyannote.audio import hatası: {str(e)}")
        
        # GPU kullanımını kontrol et
        use_gpu = torch.cuda.is_available()
        # GPU kullanılıyorsa float32 kullan, float16 ile uyumsuzluk sorunları var
        if use_gpu:
            # PyTorch'un varsayılan veri tipini float32'ye ayarla
            torch.set_default_dtype(torch.float32)
            # Belleği temizle
            torch.cuda.empty_cache()
        
        # Modeli yükle
        try:
            print(f"[{job_id}] Pyannote.audio modeli yükleniyor...")
            
            # Modeli CPU'da yükle, sonra GPU'ya taşı
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            
            # Modelin None dönüp dönmediğini kontrol et
            if diarization_pipeline is None:
                error_msg = f"[{job_id}] Pyannote.audio Pipeline.from_pretrained modeli yükleyemedi ve None döndürdü. Token: {'Token mevcut' if token else 'Token YOK'}, Model: pyannote/speaker-diarization-3.1"
                print(error_msg)
                logger.error(error_msg)
                raise ValueError("Pyannote.audio Pipeline.from_pretrained modeli yükleyemedi ve None döndürdü. Lütfen Hugging Face model erişiminizi ve ağ bağlantınızı kontrol edin.")

            print(f"[{job_id}] Pyannote.audio modeli başarıyla yüklendi")
            
            if use_gpu:
                print(f"[{job_id}] Model GPU'ya taşınıyor...")
                try:
                    # Modeli GPU'ya taşı, float32 veri tipini kullanarak
                    diarization_pipeline.to(torch.device("cuda"))
                    print(f"[{job_id}] Model GPU'ya taşındı")
                except Exception as e:
                    print(f"[{job_id}] Model GPU'ya taşınırken hata: {str(e)}")
                    print(f"[{job_id}] CPU kullanılacak")
                    use_gpu = False
                
        except Exception as e:
            print(f"[{job_id}] Pyannote.audio modeli yükleme hatası: {str(e)}")
            import traceback
            print(f"[{job_id}] Yükleme hata detayları:\n{traceback.format_exc()}")
            raise Exception(f"Pyannote.audio modeli yükleme hatası: {str(e)}")
        
        print(f"[{job_id}] Konuşmacı ayrıştırma işlemi başlıyor...")
        try:
            # İşlemi gerçekleştir
            diarization = diarization_pipeline(audio_path)
            print(f"[{job_id}] Konuşmacı ayrıştırma başarıyla tamamlandı")
            
        except Exception as e:
            print(f"[{job_id}] Konuşmacı ayrıştırma işlemi sırasında hata: {str(e)}")
            import traceback
            print(f"[{job_id}] Ayrıştırma hata detayları:\n{traceback.format_exc()}")
            
            # GPU hatası alındıysa CPU'ya geçiş yap
            if use_gpu and "cuda" in str(e).lower():
                print(f"[{job_id}] GPU hatası tespit edildi, CPU'ya geçiliyor...")
                try:
                    # Belleği temizle
                    torch.cuda.empty_cache()
                    # Modeli CPU'ya taşı
                    diarization_pipeline.to(torch.device("cpu"))
                    # İşlemi CPU'da tekrar dene
                    diarization = diarization_pipeline(audio_path)
                    print(f"[{job_id}] CPU ile konuşmacı ayrıştırma başarıyla tamamlandı")
                except Exception as cpu_e:
                    print(f"[{job_id}] CPU ile de işlem başarısız: {str(cpu_e)}")
                    raise Exception(f"Konuşmacı ayrıştırma işlemi hatası (GPU ve CPU): {str(e)}")
            else:
                raise Exception(f"Konuşmacı ayrıştırma işlemi hatası: {str(e)}")
        
        # Sonuçları listele
        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        
        print(f"[{job_id}] Konuşmacı ayrıştırma tamamlandı: {len(speakers)} segment bulundu")
        return speakers
        
    except Exception as e:
        print(f"[{job_id}] Konuşmacı ayrıştırma ana fonksiyonunda hata: {str(e)}")
        logger.error(f"[{job_id}] Konuşmacı ayrıştırma hatası: {str(e)}")
        import traceback
        logger.error(f"[{job_id}] Ayrıştırma hata detayları:\n{traceback.format_exc()}")
        # Hatayı yukarıya ilet
        raise Exception(f"Konuşmacı ayrıştırma hatası: {str(e)}") 