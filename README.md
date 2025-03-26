# Toplantı Video Analiz Aracı

Bu proje, toplantı videolarını işleyerek transkripsiyon, konuşmacı ayrıştırma ve analiz yapan bir web uygulamasıdır.

## Özellikler

- Video yükleme ve ses dosyasına dönüştürme
- OpenAI Whisper kullanarak konuşma metne çevirme
- Pyannote.Audio kullanarak konuşmacı ayrıştırma
- Konuşma metni ve konuşmacı bilgisini eşleştirme
- Toplantı katılım analizi
- İnteraktif video oynatıcı ve transkripsiyon görüntüleme

## Teknoloji Yığını

- **Backend**: Node.js, Express.js
- **Frontend**: HTML, CSS, JavaScript
- **Modeller**: Python, Flask, OpenAI Whisper, Pyannote.Audio
- **Diğer Araçlar**: FFmpeg (ses ayıklama için), Socket.IO (gerçek zamanlı iletişim için)

## Kurulum

### Gereksinimler

- Node.js (v14+)
- Python (v3.8+)
- FFmpeg
- CUDA (GPU hızlandırma için isteğe bağlı)

### Backend Kurulumu

```bash
cd backend
npm install
```

### Frontend Kurulumu

Frontend statik dosyalardan oluşur, ayrı bir kurulum gerektirmez.

### Model Servisi Kurulumu

```bash
cd model
pip install -r requirements.txt
```

Hugging Face API token'ınızı `.env` dosyasına eklemeniz gerekiyor:

```bash
cp .env.example .env
# .env dosyasını düzenleyin ve HUGGINGFACE_TOKEN değerini ekleyin
```

## Çalıştırma

1. Backend'i başlatın:

```bash
cd backend
npm start
```

2. Model servisini başlatın:

```bash
cd model
python app.py
```

3. Frontend'e erişin:
   - Backend sunucusu `/frontend` klasörüne statik dosya sunucusu olarak hizmet verir
   - Tarayıcınızda `http://localhost:3000` adresine gidin

## İş Akışı

1. Kullanıcı bir toplantı videosu yükler
2. Backend video dosyasını alır ve FFmpeg kullanarak ses dosyasına dönüştürür
3. Ses dosyası model servisine gönderilir
4. Model servisi şu işlemleri gerçekleştirir:
   - Whisper modeli ile ses transkripsiyon işlemi
   - Pyannote.Audio ile konuşmacı ayrıştırma
   - Transkripsiyon ve konuşmacı bilgilerini eşleştirme
   - Toplantı analizi
5. Sonuçlar frontend'e gönderilir ve kullanıcıya gösterilir

## Not

Bu uygulamayı kullanmak için Hugging Face hesabınızın pyannote/speaker-diarization-3.1 modeline erişim yetkisine sahip olması gerekir. Bu modeli kullanmak için kabul etmeniz gereken lisans koşulları vardır.

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen pull request göndermeden önce değişikliklerinizi test edin. 