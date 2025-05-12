# Toplantı analizi fonksiyonlarını içerir. Konuşma oranları, süre analizleri gibi temel analizleri yapar.

import logging
import json
from datetime import datetime
import os
from .topic import detect_meeting_topic
from .sentiment import analyze_sentiment

logger = logging.getLogger(__name__)

def save_analysis_results(analysis_results, output_dir="meeting_analyses"):
    """
    Analiz sonuçlarını JSON formatında kaydeder.
    Args:
        analysis_results (dict): Analiz sonuçları
        output_dir (str): Sonuçların kaydedileceği dizin
    """
    try:
        # Dizini oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Dosya adını tarih ve saat ile oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meeting_analysis_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Sonuçları JSON formatında kaydet
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=4)
        
        print(f"Analiz sonuçları kaydedildi: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Sonuçlar kaydedilirken hata oluştu: {str(e)}")
        return None

def analyze_meeting(aligned_transcript):
    try:
        print(f"Toplantı analizi başlatılıyor...")
        print(f"Analiz için {len(aligned_transcript) if isinstance(aligned_transcript, list) else 'Hatalı'} segment mevcut")
        
        # Basit analiz örneği (gerçek uygulamada daha gelişmiş olabilir)
        speakers = {}
        speaker_dialogues = {}  # Konuşmacıların diyaloglarını tutacak sözlük
        total_duration = 0
        
        for segment in aligned_transcript:
            try:
                speaker = segment["speaker"]
                duration = segment["end"] - segment["start"]
                text = segment["text"]
                total_duration += duration
                
                # Konuşmacı istatistikleri
                if speaker not in speakers:
                    speakers[speaker] = {
                        "speaking_time": 0,
                        "segments": 0,
                        "words": 0
                    }
                    speaker_dialogues[speaker] = []  # Her konuşmacı için boş diyalog listesi oluştur
                
                speakers[speaker]["speaking_time"] += duration
                speakers[speaker]["segments"] += 1
                speakers[speaker]["words"] += len(text.split())
                
                # Konuşmacının diyalogunu kaydet
                speaker_dialogues[speaker].append({
                    "text": text,
                    "start_time": segment["start"],
                    "end_time": segment["end"]
                })
                
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
        summary = f"{meeting_topic}"
        
        print(f"Analiz tamamlandı, özet: {summary}")
        
        analysis_results = {
            "summary": summary,
            "topic": meeting_topic,
            "participation": participation,
            "speaker_stats": speakers,
            "sentiment": meeting_sentiment,
            "speaker_dialogues": speaker_dialogues  # Diyalogları sonuçlara ekle
        }
        
        # Sonuçları kaydet
        save_analysis_results(analysis_results)
        
        return analysis_results
    
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