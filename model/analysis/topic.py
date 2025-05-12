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
        
        # Pegasus özetleme modelini yükle
        print("Pegasus özetleme modeli yükleniyor...")
        model_name = "google/pegasus-xsum"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Model GPU'ya taşı
        if torch.cuda.is_available():
            model = model.to("cuda")
            
        print("Pegasus modeli başarıyla yüklendi")
        
        # Modele metni ilet
        print("Toplantı konusu özeti oluşturuluyor...")
        inputs = tokenizer(truncated_text, return_tensors="pt", truncation=True)
        
        # GPU'ya taşı
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        # Özetleme için optimize edilmiş parametreler
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=6,            # Beam search için kullanılacak beam sayısı artırıldı
            min_length=75,          # Minimum özet uzunluğu artırıldı
            max_length=200,         # Maximum özet uzunluğu artırıldı
            length_penalty=2.0,     # Daha uzun özetleri teşvik et
            early_stopping=True,    # Tüm beamler EOS'a ulaştığında durdur
            no_repeat_ngram_size=3, # Kelime tekrarını önle
            do_sample=True,         # Yaratıcı özetler için sampling aktif
            top_k=50,              # Top-k sampling için parametre
            top_p=0.95,            # Nucleus sampling için parametre
            temperature=0.7         # Yaratıcılık seviyesi
        )
        
        # Tokenlardan metne çevir
        topic = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Özeti iyileştir
        topic = topic.replace(" .", ".").replace(" ,", ",").strip()
        
        print(f"Özet oluşturuldu: {topic[:100]}...")
        return topic
        
    except Exception as e:
        print(f"Özet oluşturma hatası: {str(e)}")
        import traceback
        print(f"Hata detayları:\n{traceback.format_exc()}")
        logger.error(f"Özet oluşturma hatası: {str(e)}")
        return "Toplantı konusu belirlenemedi" 