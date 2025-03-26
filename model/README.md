# Toplantı Video Analiz - Model Servisi

Bu klasör, toplantı video analiz aracının model servis kısmını içerir. Bu servis, konuşma metne çevirme (speech-to-text), konuşmacı ayrıştırma (speaker diarization) ve analiz işlemlerini gerçekleştirir.

## Kullanılan Modeller

1. **OpenAI Whisper (large-v3)**: Ses dosyalarını metin transkripsiyon için kullanılır. Whisper modelinin büyük versiyonu yüksek kaliteli transkripsiyon sağlar ve çeşitli aksanlara karşı dayanıklıdır.

2. **Pyannote.Audio (speaker-diarization-3.1)**: Konuşmacı ayrıştırma için kullanılır. Bu model, bir ses dosyasındaki farklı konuşmacıları ayırt edebilir ve her biri için zaman aralıklarını belirleyebilir.

## API Endpointleri

### POST /api/process
Ses dosyasını işlemeye başlatır.

**İstek**:
```json
{
  "audio_path": "/tam/yol/ses_dosyasi.mp3"
}
```

**Yanıt**:
```json
{
  "message": "Processing started",
  "job_id": "<job_id>"
}
```

### GET /api/status/<job_id>
Bir işin durumunu kontrol eder.

**Yanıt**:
```json
{
  "status": "processing|completed|error",
  "error": "Hata mesajı (varsa)"
}
```

### GET /api/result/<job_id>
Tamamlanan bir işin sonuçlarını alır.

**Yanıt**:
```json
{
  "status": "completed",
  "transcription": "Toplantı transkripsiyon tam metni...",
  "aligned_transcript": [
    {
      "speaker": "SPEAKER_01",
      "text": "Merhaba, toplantıya hoşgeldiniz.",
      "start": 0.0,
      "end": 2.5
    },
    ...
  ],
  "speakers": [
    {
      "speaker": "SPEAKER_01",
      "start": 0.0,
      "end": 2.5
    },
    ...
  ],
  "analysis": {
    "summary": "Toplantı özeti bilgisi...",
    "participation": {
      "SPEAKER_01": 0.65,
      "SPEAKER_02": 0.35
    },
    "speaker_stats": {
      "SPEAKER_01": {
        "speaking_time": 120.5,
        "segments": 15,
        "words": 250
      },
      ...
    },
    "sentiment": "neutral"
  }
}
```

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. Hugging Face API token ayarları:
```bash
cp .env.example .env
# .env dosyasını düzenleyip HUGGINGFACE_TOKEN değerini ekleyin
```

## Çalıştırma

```bash
python app.py
```

Varsayılan olarak, Flask sunucusu 5000 portunda çalışacaktır.

## GPU Desteği

GPU hızlandırması kullanmak için:

1. CUDA kurulu olmalıdır
2. PyTorch'un CUDA versiyonu yüklü olmalıdır
3. `.env` dosyasındaki `CUDA_VISIBLE_DEVICES` değerini uygun şekilde ayarlayın

## Notlar

- Konuşmacı ayrıştırma (speaker diarization) modeli için Hugging Face API token'ına ihtiyaç vardır
- Whisper modeli ilk kullanımda indirilecektir (yaklaşık 3GB)
- Bu servis, büyük modeller içerdiği için yeterli RAM (en az 8GB, tercihen 16GB) gerektirir 