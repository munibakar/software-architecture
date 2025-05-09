# Toplantının konusunu tespit eder. NLP modellerini kullanarak içerik analizi yapar.

import logging
import torch

logger = logging.getLogger(__name__)

def detect_meeting_topic(text, aligned_transcript=None):
    try:
        print("Toplantı konusu tespiti başlatılıyor...")
        
        # Transformers modelini import et
        try:
            print("Transformers modülünü import ediliyor...")
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            print("Transformers import edildi")
        except Exception as e:
            print(f"Transformers import hatası: {str(e)}")
            raise Exception(f"Transformers import hatası: {str(e)}")
        
        # Toplantı içeriğini hazırla
        if aligned_transcript and len(aligned_transcript) > 0:
            print("Transkripsiyon segmentleri kullanılarak içerik hazırlanıyor...")
            
            # Toplantının başlangıç kısmına daha fazla ağırlık ver (ilk %20)
            intro_ratio = 0.2
            intro_segments = aligned_transcript[:int(len(aligned_transcript) * intro_ratio)]
            intro_text = " ".join([segment["text"] for segment in intro_segments])
            
            # Toplantının sonuç kısmına daha fazla ağırlık ver (son %20)
            outro_ratio = 0.2
            outro_segments = aligned_transcript[-int(len(aligned_transcript) * outro_ratio):]
            outro_text = " ".join([segment["text"] for segment in outro_segments])
            
            # Tüm metni de ekle
            full_text = " ".join([segment["text"] for segment in aligned_transcript])
            
            # Başlangıç ve sonuç metinlerini özellikle vurgula
            prepared_text = f"Toplantı özeti: {intro_text} {full_text} Sonuç: {outro_text}"
            
        else:
            # Eğer segment bilgisi yoksa direkt metni kullan
            prepared_text = text
        
        # Metni kısaltmamız gerekebilir (model genellikle token limitine sahip)
        max_length = min(1024, len(prepared_text.split()))
        truncated_text = " ".join(prepared_text.split()[:max_length])
        print(f"Konu tespiti için metin hazırlandı, uzunluk: {len(truncated_text.split())} kelime")
        
        # GPU kontrolü
        device = 0 if torch.cuda.is_available() else -1
        print(f"Cihaz: {device}, CUDA kullanılabilir: {torch.cuda.is_available()}")
        
        # Metin özetleme modelini doğrudan yükle (daha fazla kontrol için)
        print("BART özetleme modeli yükleniyor...")
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Model GPU'ya taşı
        if torch.cuda.is_available():
            model = model.to("cuda")
            
        print("BART modeli başarıyla yüklendi")
        
        # Modele metni ilet
        print("Toplantı konusu özeti oluşturuluyor...")
        inputs = tokenizer(truncated_text, return_tensors="pt", max_length=1024, truncation=True)
        
        # GPU'ya taşı
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        # Özetleme için optimize edilmiş parametreler
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,            # Beam search için kullanılacak beam sayısı
            min_length=15,          # Minimum özet uzunluğu
            max_length=60,          # Maksimum özet uzunluğu
            length_penalty=2.0,     # Daha uzun özetleri teşvik et
            early_stopping=True,    # Tüm beamler EOS'a ulaştığında durdur
            no_repeat_ngram_size=3, # Kelime tekrarını önle
            do_sample=False         # Belirleyici çıktı için
        )
        
        # Tokenlardan metne çevir
        topic = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Sonucu biçimlendir
        topic = topic.strip()
        
        # "Toplantı" kelimesini başa ekle (eğer yoksa)
        if not topic.lower().startswith("toplantı"):
            if not topic.endswith("."):
                topic = topic + "."
            topic = f"Toplantı {topic}"
        
        print(f"Tespit edilen toplantı konusu: {topic}")
        
        return topic
    
    except Exception as e:
        print(f"Konu tespiti hatası: {str(e)}")
        import traceback
        print(f"Konu tespiti hata detayları:\n{traceback.format_exc()}")
        
        # Hata durumunda basit bir kelime sıklığı analizi yap (yedek yöntem)
        try:
            print("Basit kelime sıklığı analizi yapılıyor (yedek yöntem)...")
            stopwords = ["ve", "veya", "ile", "bu", "bir", "için", "gibi", "ben", "sen", "o", "biz", "siz", "onlar"]
            words = text.lower().split()
            filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
            
            # Kelime frekanslarını hesapla
            word_freq = {}
            for word in filtered_words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
            
            # En sık kullanılan kelimeleri bul
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            top_words = sorted_words[:5] if len(sorted_words) >= 5 else sorted_words
            
            # İlişkili kelimeleri birleştirerek konu tahmin et
            if len(top_words) > 0:
                topic_keywords = [word for word, _ in top_words]
                topic = ", ".join(topic_keywords[:3])
                return f"Toplantı şu konuları içeriyor: {topic}."
            else:
                return "Toplantı konusu belirlenemedi."
        except:
            return "Toplantı konusu belirlenemedi." 