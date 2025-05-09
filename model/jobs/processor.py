# İş süreçlerini yöneten kodları içerir. Ses analizi işlerini arkaplanda çalıştırır ve sonuçları önbellekte saklar.


import os
import logging
import time
import traceback
from threading import Thread

# Modülleri import et
from ..audio.transcription import transcribe_audio
from ..audio.diarization import diarize_audio
from ..text.alignment import align_transcription_with_speakers
from ..analysis.meeting import analyze_meeting

logger = logging.getLogger(__name__)

# Global değişkenler
results_cache = {}  # Tamamlanan işleri önbelleğe almak için

def process_job(audio_path, job_id):
    try:
        print(f"[{job_id}] İşlem başlatılıyor: {audio_path}")
        logger.info(f"[{job_id}] İşlem başlatılıyor: {audio_path}")
        
        # İşlem başladığını results_cache'e kaydet
        results_cache[job_id] = {"status": "processing"}
        print(f"[{job_id}] Durum 'processing' olarak ayarlandı")
        
        try:
            # 1. Transkripsiyon
            print(f"[{job_id}] Transkripsiyon başlatılıyor...")
            transcription, chunks = transcribe_audio(audio_path, job_id)
            print(f"[{job_id}] Transkripsiyon tamamlandı. Metin uzunluğu: {len(transcription)}, Segment sayısı: {len(chunks) if chunks else 0}")
            
            # 2. Konuşmacı ayrıştırma
            print(f"[{job_id}] Konuşmacı ayrıştırma başlatılıyor...")
            speakers = diarize_audio(audio_path, job_id)
            print(f"[{job_id}] Konuşmacı ayrıştırma tamamlandı. Segment sayısı: {len(speakers)}")
            
            # 3. Transkripsiyon ve konuşmacı bilgisini birleştir
            print(f"[{job_id}] Transkripsiyon ve konuşmacı eşleştirme başlatılıyor...")
            aligned_transcript = align_transcription_with_speakers(transcription, chunks, speakers)
            print(f"[{job_id}] Eşleştirme tamamlandı. Eşleştirilmiş segment sayısı: {len(aligned_transcript) if isinstance(aligned_transcript, list) else 'Hata'}")
            
            # 4. Toplantı analizi
            print(f"[{job_id}] Toplantı analizi başlatılıyor...")
            analysis = analyze_meeting(aligned_transcript)
            print(f"[{job_id}] Toplantı analizi tamamlandı")
            
            # Sonuçları önbelleğe al
            results_cache[job_id] = {
                "status": "completed",
                "transcription": transcription,
                "aligned_transcript": aligned_transcript,
                "speakers": speakers,
                "analysis": analysis
            }
            
            print(f"[{job_id}] Sonuçlar cache'e kaydedildi, durum 'completed' olarak ayarlandı")
            logger.info(f"[{job_id}] İşlem tamamlandı")
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"[{job_id}] İşlem sırasında hata: {str(e)}")
            print(f"[{job_id}] Hata detayları:\n{error_traceback}")
            logger.error(f"[{job_id}] İşlem hatası: {str(e)}")
            logger.error(f"[{job_id}] Hata detayları:\n{error_traceback}")
            
            # Hata durumunu cache'e kaydet
            results_cache[job_id] = {
                "status": "error",
                "error": str(e),
                "traceback": error_traceback
            }
            print(f"[{job_id}] Hata durumu cache'e kaydedildi")
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"[{job_id}] Ana işlem fonksiyonunda kritik hata: {str(e)}")
        print(f"[{job_id}] Kritik hata detayları:\n{error_traceback}")
        logger.error(f"[{job_id}] Ana işlem fonksiyonunda kritik hata: {str(e)}")
        logger.error(f"[{job_id}] Kritik hata detayları:\n{error_traceback}")
        
        try:
            # Son çare olarak hata durumunu kaydet
            results_cache[job_id] = {
                "status": "error",
                "error": f"Kritik hata: {str(e)}",
                "traceback": error_traceback
            }
            print(f"[{job_id}] Kritik hata durumu cache'e kaydedildi")
        except:
            print(f"[{job_id}] Cache'e yazma sırasında bile hata oluştu!") 