from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import numpy as np
import time
from dotenv import load_dotenv
import logging
import json
from threading import Thread

# Uygulama oluşturma
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Her istekten sonra CORS başlıklarını ekleyelim
@app.after_request
def after_request(response):
    print("İstek alındı:", request.method, request.path)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Loglama konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Çevre değişkenlerini yükle
load_dotenv()

# FFmpeg'i otomatik olarak yapılandır
try:
    backend_ffmpeg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    ffmpeg_exe_path = os.path.join(backend_ffmpeg_path, "ffmpeg.exe")
    
    if os.path.exists(ffmpeg_exe_path):
        print(f"FFmpeg bulundu: {ffmpeg_exe_path}")
        # FFmpeg'i PATH'e ekle
        os.environ["PATH"] = f"{backend_ffmpeg_path};{os.environ['PATH']}"
        print(f"FFmpeg PATH'e eklendi: {backend_ffmpeg_path}")
    else:
        print(f"FFmpeg bulunamadı: {ffmpeg_exe_path}")
except Exception as e:
    print(f"FFmpeg yapılandırma hatası: {str(e)}")

# Global değişkenler
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
results_cache = {}  # Tamamlanan işleri önbelleğe almak için

# Whisper modeli için fonksiyon
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

# Konuşmacı ayrıştırma için fonksiyon
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
        
        # Modeli yükle
        try:
            print(f"[{job_id}] Pyannote.audio modeli yükleniyor...")
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
            
            if torch.cuda.is_available():
                print(f"[{job_id}] Model GPU'ya taşınıyor...")
                diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
                print(f"[{job_id}] Model GPU'ya taşındı")
                
        except Exception as e:
            print(f"[{job_id}] Pyannote.audio modeli yükleme hatası: {str(e)}")
            import traceback
            print(f"[{job_id}] Yükleme hata detayları:\n{traceback.format_exc()}")
            raise Exception(f"Pyannote.audio modeli yükleme hatası: {str(e)}")
        
        print(f"[{job_id}] Konuşmacı ayrıştırma işlemi başlıyor...")
        try:
            diarization = diarization_pipeline(audio_path)
            print(f"[{job_id}] Konuşmacı ayrıştırma başarıyla tamamlandı")
            
        except Exception as e:
            print(f"[{job_id}] Konuşmacı ayrıştırma işlemi sırasında hata: {str(e)}")
            import traceback
            print(f"[{job_id}] Ayrıştırma hata detayları:\n{traceback.format_exc()}")
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

# Transkripsiyon ve konuşmacı ayrıştırma sonuçlarını birleştirme
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

# Toplantı analizi
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

# Toplantı konusu tespiti fonksiyonu
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
        import torch
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

# Duygu analizi fonksiyonu
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
        import torch
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

# İşleme iş parçacığı
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
            import traceback
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
        import traceback
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

# API rotaları
@app.route('/api/process', methods=['POST'])
def start_processing():
    try:
        print("POST /api/process endpoint'i çağrıldı")
        data = request.json
        print(f"Alınan veri: {data}")
        
        audio_path = data.get('audio_path')
        print(f"Ses dosyası yolu: {audio_path}")
        
        if not audio_path:
            print("HATA: Ses dosyası yolu belirtilmedi")
            return jsonify({"error": "Audio path is required"}), 400
        
        # Dosyanın varlığını ve erişilebilirliğini kontrol et
        print(f"Dosya var mı: {os.path.exists(audio_path)}")
        if os.path.exists(audio_path):
            print(f"Dosya boyutu: {os.path.getsize(audio_path)} bytes")
            print(f"Dosya okunabilir mi: {os.access(audio_path, os.R_OK)}")
        
        if not os.path.exists(audio_path):
            print(f"HATA: Ses dosyası bulunamadı: {audio_path}")
            return jsonify({"error": f"Audio file not found: {audio_path}"}), 404
        
        job_id = str(int(time.time()))
        print(f"Oluşturulan job_id: {job_id}")
        
        # İşlemi arka planda başlat
        print("Arka plan işlemi başlatılıyor...")
        thread = Thread(target=process_job, args=(audio_path, job_id))
        thread.daemon = True
        thread.start()
        
        print(f"İşlem başlatıldı, job_id: {job_id}")
        return jsonify({
            "message": "Processing started",
            "job_id": job_id
        })
        
    except Exception as e:
        print(f"API hatası: {str(e)}")
        logger.error(f"API hatası: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    print(f"GET /api/status/{job_id} endpoint'i çağrıldı")
    
    if job_id not in results_cache:
        print(f"HATA: Job ID bulunamadı: {job_id}")
        return jsonify({"status": "not_found"}), 404
    
    print(f"Durum yanıtı: {results_cache[job_id]}")
    return jsonify(results_cache[job_id])

@app.route('/api/result/<job_id>', methods=['GET'])
def get_job_result(job_id):
    print(f"GET /api/result/{job_id} endpoint'i çağrıldı")
    
    if job_id not in results_cache:
        print(f"HATA: Job ID bulunamadı: {job_id}")
        return jsonify({"error": "Job not found"}), 404
    
    job_result = results_cache[job_id]
    
    if job_result["status"] != "completed":
        print(f"HATA: İşlem henüz tamamlanmadı. Durum: {job_result['status']}")
        return jsonify({
            "status": job_result["status"],
            "error": job_result.get("error", "Job is still processing")
        }), 400
    
    print(f"Sonuç başarıyla döndürüldü")
    return jsonify(job_result)

# İlave test endpoint'i
@app.route('/api/test', methods=['GET', 'POST'])
def test_api():
    print(f"Test API çağrısı: {request.method}")
    if request.method == 'POST':
        print(f"POST veri: {request.json}")
    
    return jsonify({
        "status": "success",
        "message": "Model servisi çalışıyor!",
        "time": time.time()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 