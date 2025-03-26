# Toplantı Video Analiz - Frontend

Bu klasör, toplantı video analiz aracının frontend kısmını içerir.

## Yapı

Frontend, saf HTML, CSS ve JavaScript kullanılarak oluşturulmuştur. Herhangi bir build adımı veya karmaşık kurulum gerektirmez.

### Dosyalar

- `index.html`: Ana HTML yapısı
- `styles.css`: Görünüm stilleri
- `app.js`: Uygulama mantığı ve API iletişimi

## Özellikler

- Video dosyası yükleme
- Gerçek zamanlı işleme durum güncellemeleri (Socket.IO)
- Video oynatıcı
- Transkripsiyon görüntüleme (konuşmacı etiketleriyle)
- Toplantı analizi görüntüleme:
  - Toplantı özeti
  - Konuşmacı katılım oranları (grafik olarak)

## Kullanılan Kütüphaneler

- Bootstrap 5: UI bileşenleri ve düzen
- Chart.js: Katılım oranlarını görselleştirmek için
- Socket.IO: Gerçek zamanlı iletişim için

## Kullanım

Frontend, backend tarafından static dosyalar olarak sunulur. Herhangi bir ayrı sunucuya gerek yoktur. 

1. Backend'i başlatın
2. Tarayıcınızdan `http://localhost:3000` adresine gidin

## Ekran Görüntüleri

### Ana Sayfa
Uygulama ilk açıldığında, kullanıcıya video yükleme formu gösterilir.

### İşleme Durumu
Bir video yüklendikten sonra, uygulama işleme durumu hakkında gerçek zamanlı güncellemeler sağlar.

### Sonuçlar
İşleme tamamlandıktan sonra, kullanıcı transkripsiyon ve analiz sonuçlarını görüntüleyebilir.

## Tarayıcı Uyumluluğu

Uygulama modern tarayıcılarda (Chrome, Firefox, Safari, Edge) çalışacak şekilde tasarlanmıştır. 