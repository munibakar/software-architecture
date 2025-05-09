#  Duygusal analiz yapar. Konuşmanın olumlu, olumsuz veya nötr olduğunu belirler.


import logging
import torch

logger = logging.getLogger(__name__)

def analyze_sentiment(transcript):
    try:
        print("Duygu analizi başlatılıyor...")
        
        # Tüm metni birleştir
        all_text = " ".join([segment["text"] for segment in transcript])
        
        # Modeli import et
        try:
            print("Transformers modülünü import ediliyor...")
            from transformers import pipeline
            print("Transformers import edildi")
        except Exception as e:
            print(f"Transformers import hatası: {str(e)}")
            raise Exception(f"Transformers import hatası: {str(e)}")
        
        # GPU kontrolü
        device = 0 if torch.cuda.is_available() else -1
        print(f"Cihaz: {device}, CUDA kullanılabilir: {torch.cuda.is_available()}")
        
        # Duygu analizi modeli
        print("Duygu analizi modeli yükleniyor...")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english", 
            device=device
        )
        print("Duygu analizi modeli başarıyla yüklendi")
        
        # Metni uygun parçalara böl (model genellikle token limitine sahip)
        chunk_size = 500  # distilbert için yaklaşık 500 kelimelik parçalar uygun
        text_chunks = []
        words = all_text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            text_chunks.append(chunk)
        
        print(f"Metin {len(text_chunks)} parçaya bölündü")
        
        # Her parça için duygu analizi yap
        results = []
        for chunk in text_chunks:
            result = sentiment_analyzer(chunk)
            results.append(result[0])
        
        # Sonuçları değerlendir
        positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
        total_chunks = len(results)
        positive_ratio = positive_count / total_chunks if total_chunks > 0 else 0
        
        # Genel duygu durumunu belirle
        if positive_ratio > 0.6:
            sentiment = "positive"
            description = "olumlu ve yapıcı"
        elif positive_ratio < 0.4:
            sentiment = "negative"
            description = "gergin ve problemli"
        else:
            sentiment = "neutral"
            description = "nötr"
        
        print(f"Duygu analizi tamamlandı: {sentiment} ({positive_ratio:.2f})")
        
        return {
            "overall": sentiment,
            "description": description,
            "score": positive_ratio
        }
    
    except Exception as e:
        print(f"Duygu analizi hatası: {str(e)}")
        import traceback
        print(f"Duygu analizi hata detayları:\n{traceback.format_exc()}")
        
        # Hata durumunda basit bir sözlük temelli duygu analizi yap
        try:
            print("Basit sözlük temelli duygu analizi yapılıyor (yedek yöntem)...")
            
            # Duygu belirten kelimeleri tanımla
            positive_words = ["teşekkür", "harika", "mükemmel", "iyi", "güzel", "başarı", "başarılı", "mutlu", 
                            "sevindirici", "olumlu", "hayal", "umut", "heyecan", "destekli", "eğlenceli"]
            
            negative_words = ["maalesef", "kötü", "sorun", "problem", "hata", "yanlış", "olumsuz", "başarısız", 
                            "üzgün", "kaygı", "endişe", "korku", "öfke", "sinir", "gergin", "stres"]
            
            # Duygu puanları
            total_score = 0
            word_count = 0
            
            # Her segment için duygu analizi yap
            for segment in transcript:
                text = segment["text"].lower()
                words = text.split()
                
                for word in words:
                    word_count += 1
                    if word in positive_words:
                        total_score += 1
                    elif word in negative_words:
                        total_score -= 1
            
            # Ortalama duygu skoru
            avg_score = total_score / word_count if word_count > 0 else 0
            
            # Duygu durumu belirleme
            if avg_score > 0.05:
                sentiment = "positive"
                description = "olumlu ve yapıcı"
            elif avg_score < -0.05:
                sentiment = "negative"
                description = "gergin ve problemli"
            else:
                sentiment = "neutral"
                description = "nötr"
            
            return {
                "overall": sentiment,
                "description": description,
                "score": avg_score
            }
        except Exception as e:
            print(f"Basit duygu analizi de başarısız oldu: {str(e)}")
            return {"overall": "unknown", "description": "belirsiz", "score": 0}
    
    except Exception as e:
        print(f"Duygu analizi hatası: {str(e)}")
        return {"overall": "unknown", "description": "belirsiz", "score": 0} 