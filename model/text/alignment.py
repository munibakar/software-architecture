# Transkripsiyon ve konuşmacı verilerini eşleştiren fonksiyonları içerir. Her metin parçasını ilgili konuşmacıyla ilişkilendirir.

import logging

logger = logging.getLogger(__name__)

def align_transcription_with_speakers(transcription, chunks, speakers):
    try:
        print(f"Transkripsiyon ve konuşmacı birleştirme başlatılıyor")
        print(f"Transkripsiyon: {transcription[:100]}...")
        print(f"Chunk sayısı: {len(chunks) if chunks else 0}")
        print(f"Konuşmacı segment sayısı: {len(speakers)}")
        
        if not chunks or not speakers:
            print(f"Chunk veya speaker verisi eksik. Birleştirme yapılamıyor.")
            if not chunks and transcription:
                # Eğer chunk yok ama transkripsiyon varsa, tüm transkripsiyon için default speaker ata
                print(f"Chunk yok ama transkripsiyon var. Tüm metni SPEAKER_01'e atıyorum.")
                return [{
                    "speaker": "SPEAKER_01",
                    "text": transcription,
                    "start": 0.0,
                    "end": 60.0  # Varsayılan bir süre
                }]
            return []
        
        aligned_text = []
        
        # Eğer transkripsiyon boş ama chunk varsa işleme devam et
        if not transcription and chunks:
            print(f"Transkripsiyon metni boş ama {len(chunks)} chunk mevcut.")
        
        for chunk in chunks:
            try:
                chunk_start = chunk["timestamp"][0]
                chunk_end = chunk["timestamp"][1]
                chunk_text = chunk["text"].strip()
                
                print(f"Segment işleniyor: {chunk_start:.2f}-{chunk_end:.2f}, Metin: {chunk_text[:30]}...")
                
                # Belirli bir zaman aralığıyla en çok örtüşen konuşmacıyı bul
                max_overlap = 0
                best_speaker = None
                
                for speaker_segment in speakers:
                    s_start = speaker_segment["start"]
                    s_end = speaker_segment["end"]
                    
                    # Zaman aralıkları arasındaki örtüşmeyi hesapla
                    overlap_start = max(chunk_start, s_start)
                    overlap_end = min(chunk_end, s_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = speaker_segment["speaker"]
                
                # Eğer eşleşme bulunamasa bile, varsayılan olarak en yakın konuşmacıyı bul
                if not best_speaker and speakers:
                    # En yakın konuşmacıyı bul (zaman mesafesine göre)
                    min_distance = float('inf')
                    for speaker_segment in speakers:
                        s_start = speaker_segment["start"]
                        s_end = speaker_segment["end"]
                        
                        # Segment öncesindeyse başlangıç mesafesini al
                        if chunk_end <= s_start:
                            distance = s_start - chunk_end
                        # Segment sonrasındaysa bitiş mesafesini al
                        elif chunk_start >= s_end:
                            distance = chunk_start - s_end
                        # Örtüşme varsa mesafe 0
                        else:
                            distance = 0
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_speaker = speaker_segment["speaker"]
                    
                    print(f"Direkt eşleşme bulunamadı, en yakın konuşmacı: {best_speaker}, Mesafe: {min_distance:.2f}s")
                
                # Varsayılan olarak ilk konuşmacıyı kullan eğer hala eşleşme yoksa
                if not best_speaker and speakers:
                    best_speaker = speakers[0]["speaker"]
                    print(f"Yakınlık eşleşmesi de bulunamadı, varsayılan konuşmacı: {best_speaker}")
                
                # Metin boş olsa bile segment ekle (sadece zaman bilgisi için)
                if best_speaker:
                    text_to_use = chunk_text if chunk_text else "[sessiz segment]"
                    aligned_text.append({
                        "speaker": best_speaker,
                        "text": text_to_use,
                        "start": chunk_start,
                        "end": chunk_end
                    })
                    print(f"Segment eklendi: {best_speaker}, Metin: {text_to_use[:20]}...")
                else:
                    print(f"Eşleşme bulunamadı, segment atlanıyor")
            except Exception as e:
                print(f"Segment işleme hatası: {str(e)}")
                import traceback
                print(f"Segment hata detayları:\n{traceback.format_exc()}")
        
        # Sonuç boşsa ve tam transkripsiyon varsa, tüm metni tek bir segmente dönüştür
        if not aligned_text and transcription:
            print(f"Eşleştirilmiş segment oluşturulamadı, tüm transkripsiyon metni tek segment olarak ekleniyor")
            aligned_text.append({
                "speaker": "SPEAKER_01",
                "text": transcription,
                "start": 0.0,
                "end": 60.0  # Varsayılan bir süre
            })
        
        print(f"Birleştirme tamamlandı. Toplam {len(aligned_text)} segment oluşturuldu.")
        return aligned_text
    
    except Exception as e:
        print(f"Transkripsiyon ve konuşmacı birleştirme hatası: {str(e)}")
        import traceback
        print(f"Birleştirme hata detayları:\n{traceback.format_exc()}")
        # Hata durumunda transkripsiyon varsa, onu tek bir segment olarak döndür
        if transcription:
            return [{
                "speaker": "SPEAKER_01",
                "text": transcription,
                "start": 0.0,
                "end": len(transcription.split()) / 2.0  # Yaklaşık olarak saniye cinsinden süre (kelime başına 0.5 saniye)
            }]
        return [] 