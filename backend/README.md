# Toplantı Video Analiz - Backend

Bu klasör, toplantı video analiz aracının backend kısmını içerir.

## API Endpointleri

### POST /api/upload
Video yükleme endpointi.

**Parametre**:
- `video`: Yüklenecek video dosyası (multipart/form-data)

**Yanıt**:
```json
{
  "message": "Video başarıyla yüklendi ve işleme alındı",
  "videoId": "<video_id>",
  "videoPath": "/uploads/<video_filename>",
  "audioPath": "/audio/<audio_filename>"
}
```

### POST /api/process
Ses dosyasını işlemek için model servisine istek gönderir.

**İstek**:
```json
{
  "audioPath": "/audio/<audio_filename>"
}
```

**Yanıt**:
```json
{
  "message": "İşlem model servise iletildi",
  "jobId": "<job_id>"
}
```

### GET /api/result/:jobId
İşlem sonuçlarını alır.

**Yanıt**:
```json
{
  "status": "completed",
  "transcription": "Toplantı transkripsiyon metni...",
  "speakers": [
    { "id": "SPEAKER_01", "segments": [{"start": 0, "end": 10}] }
  ],
  "analysis": {
    "summary": "Toplantı özeti...",
    "sentiment": "neutral",
    "participation": {
      "SPEAKER_01": 0.6,
      "SPEAKER_02": 0.4
    }
  }
}
```

## Socket.IO Olayları

Sunucu şu olayları yayınlar:

- `processingUpdate`: İşlem durumunu bildirir
  ```json
  {
    "status": "started|audioExtracted|modelProcessing|completed|error",
    "message": "Durum açıklaması...",
    "audioPath": "/audio/<audio_filename>",
    "videoPath": "/uploads/<video_filename>"
  }
  ```

## Klasör Yapısı

- `uploads/`: Yüklenen video dosyalarının saklandığı klasör
- `audio/`: Video'lardan çıkarılan ses dosyalarının saklandığı klasör

## Ortam Değişkenleri

- `PORT`: Sunucunun çalışacağı port (Varsayılan: 3000) 