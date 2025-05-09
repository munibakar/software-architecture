# Toplantı analizi fonksiyonlarını içerir. Konuşma oranları, süre analizleri gibi temel analizleri yapar.

import logging
from .topic import detect_meeting_topic
from .sentiment import analyze_sentiment

logger = logging.getLogger(__name__)

def analyze_meeting(aligned_transcript):
    try:
        print(f"Toplantı analizi başlatılıyor...")
        print(f"Analiz için {len(aligned_transcript) if isinstance(aligned_transcript, list) else 'Hatalı'} segment mevcut")
        
        # Basit analiz örneği (gerçek uygulamada daha gelişmiş olabilir)
        speakers = {}
        total_duration = 0
        
        for segment in aligned_transcript:
            try:
                speaker = segment["speaker"]
                duration = segment["end"] - segment["start"]
                total_duration += duration
                
                if speaker not in speakers:
                    speakers[speaker] = {
                        "speaking_time": 0,
                        "segments": 0,
                        "words": 0
                    }
                
                speakers[speaker]["speaking_time"] += duration
                speakers[speaker]["segments"] += 1
                speakers[speaker]["words"] += len(segment["text"].split())
            except Exception as e:
                print(f"Segment analiz hatası: {str(e)}")
        
        # Konuşma oranlarını hesapla
        participation = {}
        print(f"Konuşmacı sayısı: {len(speakers)}")
        print(f"Toplam konuşma süresi: {total_duration:.2f} saniye")
        
        for speaker, stats in speakers.items():
            participation[speaker] = stats["speaking_time"] / total_duration if total_duration > 0 else 0
            print(f"Konuşmacı {speaker}: {stats['speaking_time']:.2f}s ({participation[speaker]*100:.1f}%)")
        
        # Tüm metni birleştir
        full_text = " ".join([segment["text"] for segment in aligned_transcript])
        print(f"Toplam metin uzunluğu: {len(full_text)} karakter")
        
        # Toplantı konusu tespiti - segmentleri de geçirerek çağır
        meeting_topic = detect_meeting_topic(full_text, aligned_transcript)
        print(f"Tespit edilen toplantı konusu: {meeting_topic}")
        
        # Duygu analizi
        meeting_sentiment = analyze_sentiment(aligned_transcript)
        print(f"Toplantı duygu analizi: {meeting_sentiment['overall']}")
        
        # Özet oluştur
        summary = f"{meeting_topic} Toplantı genel olarak {meeting_sentiment['description']} bir şekilde geçmiştir."
        
        print(f"Analiz tamamlandı, özet: {summary}")
        return {
            "summary": summary,
            "topic": meeting_topic,
            "participation": participation,
            "speaker_stats": speakers,
            "sentiment": meeting_sentiment
        }
    
    except Exception as e:
        print(f"Analiz hatası: {str(e)}")
        import traceback
        print(f"Analiz hata detayları:\n{traceback.format_exc()}")
        logger.error(f"Analiz hatası: {str(e)}")
        return {
            "summary": "Analiz sırasında hata oluştu.",
            "topic": "Toplantı konusu belirlenemedi",
            "participation": {},
            "sentiment": {"overall": "unknown", "description": "belirsiz"}
        } 