# Ses dosyalarını metne çeviren fonksiyonu içerir. Whisper modelini kullanarak konuşmaları yazıya döker.


import os
import torch
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path, job_id):
    try:
        logger.info(f"[{job_id}] Transkripsiyon başlatılıyor: {audio_path}")
        print(f"[{job_id}] Transkripsiyon için ses dosyası: {audio_path}")
        print(f"[{job_id}] Dosya var mı: {os.path.exists(audio_path)}")
        
        # GPU durumunu kontrol et
        try:
            print(f"[{job_id}] GPU kontrolü yapılıyor...")
            print(f"[{job_id}] CUDA kullanılabilir mi: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[{job_id}] Kullanılabilir GPU sayısı: {torch.cuda.device_count()}")
                print(f"[{job_id}] Aktif GPU: {torch.cuda.get_device_name(0)}")
                # GPU için belleği temizle
                torch.cuda.empty_cache()
                # Veri tipini float32 olarak ayarla (float16 sorunlara neden oluyor)
                torch.set_default_dtype(torch.float32)
        except Exception as e:
            print(f"[{job_id}] GPU kontrolü sırasında hata: {str(e)}")
        
        # Modeli yükle
        print(f"[{job_id}] Whisper modeli yükleniyor...")
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"[{job_id}] Cihaz: {device}")
            
            model_id = "openai/whisper-large-v3"
            
            # Veri tipini belirle - tüm modeller için float32 kullan
            dtype = torch.float32
            
            # Model ve processor'ı yükle
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(device)
            
            processor = AutoProcessor.from_pretrained(model_id)
            
            transcriber = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                device=device,
                torch_dtype=dtype  # Aynı veri tipini pipeline'da da belirt
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
            )
            print(f"[{job_id}] Transkripsiyon başarıyla tamamlandı")
            
        except Exception as e:
            print(f"[{job_id}] Transkripsiyon işlemi sırasında hata: {str(e)}")
            import traceback
            print(f"[{job_id}] Transkripsiyon hata detayları:\n{traceback.format_exc()}")
            
            # GPU hatası alındıysa CPU'ya geçiş yap
            if torch.cuda.is_available() and "cuda" in str(e).lower():
                print(f"[{job_id}] GPU hatası tespit edildi, CPU'ya geçiliyor...")
                try:
                    # Belleği temizle
                    torch.cuda.empty_cache()
                    # Modeli CPU'ya taşı
                    model.to("cpu")
                    # Pipeline'ı yeniden oluştur
                    transcriber = pipeline(
                        "automatic-speech-recognition",
                        model=model,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                        chunk_length_s=30,
                        device="cpu"
                    )
                    # İşlemi CPU'da tekrar dene
                    result = transcriber(
                        audio_path,
                        return_timestamps=True,
                    )
                    print(f"[{job_id}] CPU ile transkripsiyon başarıyla tamamlandı")
                except Exception as cpu_e:
                    print(f"[{job_id}] CPU ile de işlem başarısız: {str(cpu_e)}")
                    raise Exception(f"Transkripsiyon işlemi hatası (GPU ve CPU): {str(e)}")
            else:
                raise Exception(f"Transkripsiyon işlemi hatası: {str(e)}")
        
        return result["text"], result.get("chunks", [])
        
    except Exception as e:
        print(f"[{job_id}] Transkripsiyon ana fonksiyonunda hata: {str(e)}")
        logger.error(f"[{job_id}] Transkripsiyon hatası: {str(e)}")
        import traceback
        logger.error(f"[{job_id}] Transkripsiyon hata detayları:\n{traceback.format_exc()}")
        return f"Transkripsiyon hatası: {str(e)}", [] 