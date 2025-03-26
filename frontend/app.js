// API Endpoint Sabitleri
const API_BASE_URL = 'http://localhost:3000';
const SOCKET_URL = 'http://localhost:3000';

// Global değişkenler
let currentVideoPath = null;
let currentAudioPath = null;
let currentJobId = null;
let socket = null;
let participationChart = null;

// DOM yüklendikten sonra çalış
document.addEventListener('DOMContentLoaded', () => {
    // Socket.io bağlantısı kurma
    setupSocketConnection();

    // Form submission olayını dinle
    const uploadForm = document.getElementById('uploadForm');
    uploadForm.addEventListener('submit', handleFormSubmit);

    // Sayfa yüklendiğinde veya kapatıldığında bağlantıyı kapat
    window.addEventListener('beforeunload', () => {
        if (socket) socket.disconnect();
    });
});

// Socket.io bağlantısı kurma
function setupSocketConnection() {
    socket = io(SOCKET_URL);

    socket.on('connect', () => {
        console.log('Socket.io bağlantısı kuruldu');
    });

    socket.on('disconnect', () => {
        console.log('Socket.io bağlantısı kesildi');
    });

    socket.on('processingUpdate', handleProcessingUpdate);
}

// Form gönderimi
async function handleFormSubmit(e) {
    e.preventDefault();

    const fileInput = document.getElementById('videoFile');
    if (!fileInput.files.length) {
        addStatusMessage('error', 'Lütfen bir video dosyası seçin');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('video', file);

    // UI'yı güncelle - başlangıç durumu
    document.getElementById('uploadBtn').disabled = true;
    document.getElementById('uploadProgress').classList.remove('d-none');
    document.getElementById('processingCard').classList.remove('d-none');
    addStatusMessage('info', 'Video yükleme hazırlanıyor...');

    try {
        // Video yüklemeyi başlat
        const uploadResponse = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData,
            // Upload progress takibi
            onUploadProgress: (progressEvent) => {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                document.querySelector('#uploadProgress .progress-bar').style.width = `${percentCompleted}%`;
                document.querySelector('#uploadProgress .progress-bar').textContent = `${percentCompleted}%`;
            }
        });

        if (!uploadResponse.ok) {
            throw new Error(`HTTP error! status: ${uploadResponse.status}`);
        }

        const uploadResult = await uploadResponse.json();
        
        // Yükleme tamamlandı, işleme başlatılıyor
        currentVideoPath = uploadResult.videoPath;
        currentAudioPath = uploadResult.audioPath;
        
        addStatusMessage('success', 'Video başarıyla yüklendi');
        addStatusMessage('info', 'Ses dosyası hazırlanıyor...');
        
        // Video oynatıcıyı göster
        displayVideo(API_BASE_URL + currentVideoPath);
        
        // Modele işleme isteği gönder
        setTimeout(async () => {
            try {
                const processResponse = await fetch(`${API_BASE_URL}/api/process`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ audioPath: currentAudioPath })
                });
                
                if (!processResponse.ok) {
                    throw new Error(`HTTP error! status: ${processResponse.status}`);
                }
                
                const processResult = await processResponse.json();
                currentJobId = processResult.jobId;
                
                addStatusMessage('info', 'Model işlemesi başlatıldı, bu işlem birkaç dakika sürebilir...');
                
                // Düzenli sonuç kontrolü başlat
                pollProcessingResult(currentJobId);
                
            } catch (error) {
                console.error('Model işleme hatası:', error);
                addStatusMessage('error', 'Model işleme hatası: ' + error.message);
                resetProcessingUI();
            }
        }, 2000);
        
    } catch (error) {
        console.error('Yükleme hatası:', error);
        addStatusMessage('error', 'Yükleme hatası: ' + error.message);
        resetProcessingUI();
    }
}

// İşleme durumu güncelleme
function handleProcessingUpdate(data) {
    console.log('İşleme güncelleme:', data);
    
    switch (data.status) {
        case 'started':
            addStatusMessage('info', data.message);
            break;
        case 'audioExtracted':
            addStatusMessage('success', data.message);
            break;
        case 'modelProcessing':
            addStatusMessage('info', data.message);
            break;
        case 'completed':
            addStatusMessage('success', 'İşlem tamamlandı!');
            // Sonuçları göster
            if (currentJobId) {
                fetchAndDisplayResults(currentJobId);
            }
            break;
        case 'error':
            addStatusMessage('error', data.message);
            resetProcessingUI();
            break;
    }
}

// Durum mesajı ekleme
function addStatusMessage(type, message) {
    const statusMessages = document.getElementById('statusMessages');
    const iconClass = {
        info: 'bi-info-circle',
        success: 'bi-check-circle',
        warning: 'bi-exclamation-triangle',
        error: 'bi-x-circle'
    };
    
    const messageHtml = `
        <div class="list-group-item list-group-item-${type}">
            <i class="bi ${iconClass[type] || 'bi-info-circle'}"></i> ${message}
        </div>
    `;
    
    statusMessages.innerHTML += messageHtml;
    statusMessages.scrollTop = statusMessages.scrollHeight;
}

// Video görüntüleme
function displayVideo(videoUrl) {
    const videoPlayerCard = document.getElementById('videoPlayerCard');
    const videoPlayer = document.getElementById('videoPlayer');
    const videoSource = videoPlayer.querySelector('source');
    
    videoSource.src = videoUrl;
    videoPlayer.load();
    videoPlayerCard.classList.remove('d-none');
}

// Sonuç kontrolü (polling)
function pollProcessingResult(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/status/${jobId}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                addStatusMessage('success', 'İşlem tamamlandı!');
                fetchAndDisplayResults(jobId);
            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                addStatusMessage('error', `İşlem hatası: ${data.error}`);
                resetProcessingUI();
            }
            
        } catch (error) {
            console.error('Durum kontrolü hatası:', error);
            // Hata durumunda interval'i durdurma - servis bağlantısı kesilmiş olabilir
            if (error.message.includes('Failed to fetch')) {
                clearInterval(pollInterval);
                addStatusMessage('error', 'Sunucu bağlantısı kesildi.');
                resetProcessingUI();
            }
        }
    }, 5000); // 5 saniyede bir kontrol et
}

// Sonuçları getir ve göster
async function fetchAndDisplayResults(jobId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/result/${jobId}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Sonuç alınamadı: ${errorData.error || response.statusText}`);
        }
        
        const data = await response.json();
        
        // Transkripsiyon verisini loglama
        console.log('API yanıtı:', data);
        console.log('Transkripsiyon segmentleri:', data.aligned_transcript ? data.aligned_transcript.length : 0);
        console.log('Tam transkripsiyon:', data.transcription ? data.transcription.substring(0, 100) + '...' : 'Yok');
        
        // Sonuç boşsa uyarı göster
        if (!data.aligned_transcript || data.aligned_transcript.length === 0) {
            addStatusMessage('warning', 'Transkripsiyon segmenti bulunamadı. Ses dosyasını kontrol edin.');
        }
        
        if (!data.transcription) {
            addStatusMessage('warning', 'Tam transkripsiyon metni bulunamadı.');
        }
        
        // Transkripsiyon göster
        displayTranscription(data.aligned_transcript, data.transcription);
        
        // Analiz göster
        displayAnalysis(data.analysis);
        
        // İşleme UI'ını temizle
        document.getElementById('processingSpinner').classList.add('d-none');
        
        // Sonuç bölümünü göster
        document.getElementById('resultsSection').classList.remove('d-none');
        
    } catch (error) {
        console.error('Sonuç getirme hatası:', error);
        addStatusMessage('error', 'Sonuç getirme hatası: ' + error.message);
        
        // Hata durumunda işleme UI'ını temizle ve sonuç bölümünü gizli tut
        document.getElementById('processingSpinner').classList.add('d-none');
    }
}

// Transkripsiyon gösterme
function displayTranscription(alignedTranscript, fullTranscription) {
    const transcriptionContent = document.getElementById('transcriptionContent');
    
    if ((!alignedTranscript || alignedTranscript.length === 0) && !fullTranscription) {
        transcriptionContent.innerHTML = '<div class="alert alert-warning">Transkripsiyon bulunamadı.</div>';
        return;
    }
    
    let html = '';
    
    // Önce tam transkripsiyon metnini göster
    if (fullTranscription) {
        html += `
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0">Tam Transkripsiyon Metni</h5>
                </div>
                <div class="card-body">
                    <div class="full-transcript">
                        ${processFullTranscription(fullTranscription)}
                    </div>
                </div>
            </div>
        `;
    }
    
    // Sonra konuşmacı bazlı transkripsiyon segmentlerini göster
    if (alignedTranscript && alignedTranscript.length > 0) {
        // Transkripsiyon kontrolü ve sıralama
        const sortedTranscript = [...alignedTranscript].sort((a, b) => a.start - b.start);
        
        // Konuşmacı listesini çıkar
        const speakers = [...new Set(sortedTranscript.map(item => item.speaker))];
        
        html += `
            <div class="card mb-4">
                <div class="card-header">
                    <div class="d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Konuşmacı Bazlı Transkripsiyon</h5>
                        <div class="speaker-filter btn-group btn-group-sm" role="group">
                            <button type="button" class="btn btn-outline-primary active" data-speaker="all">Tümü (${sortedTranscript.length})</button>
                            ${speakers.map(speaker => {
                                const count = sortedTranscript.filter(s => s.speaker === speaker).length;
                                // Konuşmacı ID'sinden sayı çıkar (SPEAKER_01 -> 1)
                                const speakerNum = speaker.replace('SPEAKER_', '').padStart(2, '0');
                                const speakerClass = `speaker-${speakerNum % 5 + 1}`;
                                return `<button type="button" class="btn btn-outline-primary ${speakerClass}" data-speaker="${speaker}">${speaker} (${count})</button>`;
                            }).join('')}
                        </div>
                    </div>
                </div>
                <div class="card-body transcript-segments">
                    <div class="timeline">
        `;
        
        // Her segment için
        sortedTranscript.forEach((segment, index) => {
            // Konuşmacı ID'sinden sayı çıkar (SPEAKER_01 -> 1)
            const speakerNum = segment.speaker.replace('SPEAKER_', '').padStart(2, '0');
            const speakerClass = `speaker-${speakerNum % 5 + 1}`;
            
            const startTime = formatTime(segment.start);
            const endTime = formatTime(segment.end);
            
            html += `
                <div class="transcript-item ${speakerClass}" data-speaker="${segment.speaker}" data-start="${segment.start}" data-end="${segment.end}">
                    <div class="d-flex align-items-start">
                        <div class="transcript-indicator"></div>
                        <div class="transcript-content">
                            <div class="d-flex justify-content-between">
                                <div class="speaker">${segment.speaker}</div>
                                <div class="timestamp">${startTime} - ${endTime}</div>
                            </div>
                            <div class="text">${segment.text}</div>
                        </div>
                    </div>
                </div>
            `;
            
            // Eğer son segment değilse ve farklı bir konuşmacıya geçiliyorsa, ayraç ekle
            if (index < sortedTranscript.length - 1 && segment.speaker !== sortedTranscript[index + 1].speaker) {
                html += `<div class="speaker-divider"></div>`;
            }
        });
        
        html += `
                    </div>
                </div>
            </div>
        `;
    }
    
    transcriptionContent.innerHTML = html;
    
    // Konuşmacı filtreleme butonlarına tıklama olayı
    document.querySelectorAll('.speaker-filter button').forEach(button => {
        button.addEventListener('click', () => {
            // Aktif sınıfını ata
            document.querySelectorAll('.speaker-filter button').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            const speaker = button.getAttribute('data-speaker');
            
            // Segmentleri filtrele
            document.querySelectorAll('.transcript-item').forEach(item => {
                if (speaker === 'all' || item.getAttribute('data-speaker') === speaker) {
                    item.style.display = 'block';
                } else {
                    item.style.display = 'none';
                }
            });
            
            // Ayraçları da güncelle - görünür konuşmacılar arasında ayraç olmalı
            const visibleItems = [...document.querySelectorAll('.transcript-item')].filter(
                item => item.style.display !== 'none'
            );
            
            document.querySelectorAll('.speaker-divider').forEach(divider => {
                divider.style.display = 'none';
            });
            
            // Görünür öğeler arasında konuşmacı değişimlerinde ayraçları göster
            for (let i = 0; i < visibleItems.length - 1; i++) {
                const currentSpeaker = visibleItems[i].getAttribute('data-speaker');
                const nextSpeaker = visibleItems[i + 1].getAttribute('data-speaker');
                
                if (currentSpeaker !== nextSpeaker) {
                    const nextDivider = visibleItems[i].nextElementSibling;
                    if (nextDivider && nextDivider.classList.contains('speaker-divider')) {
                        nextDivider.style.display = 'block';
                    }
                }
            }
        });
    });
    
    // Transkripsiyon öğelerine tıklama olayı
    document.querySelectorAll('.transcript-item').forEach(item => {
        item.addEventListener('click', () => {
            const start = parseFloat(item.getAttribute('data-start'));
            jumpToVideoTime(start);
            
            // Aktif sınıfını ata
            document.querySelectorAll('.transcript-item').forEach(el => el.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

// Tam transkripsiyon metnini işle
function processFullTranscription(text) {
    if (!text) return '';
    
    // Satır sonlarına göre böl
    let paragraphs = text.split('\n');
    
    // Boş satırları kaldır ve HTML olarak işle
    paragraphs = paragraphs.filter(p => p.trim() !== '');
    
    // Paragrafları cümlelere bölerek daha okunabilir hale getir
    let html = '';
    
    paragraphs.forEach(paragraph => {
        // Nokta, soru işareti ve ünlem işaretinden sonra cümlelere böl
        const sentences = paragraph.split(/(?<=[.!?])\s+/);
        
        html += '<p>';
        sentences.forEach(sentence => {
            if (sentence.trim() !== '') {
                html += `<span class="sentence">${sentence.trim()}</span> `;
            }
        });
        html += '</p>';
    });
    
    return html;
}

// Analiz gösterme
function displayAnalysis(analysis) {
    // Özet
    document.getElementById('meetingSummary').textContent = analysis.summary;
    
    // Katılım grafiği
    createParticipationChart(analysis.participation);
}

// Katılım grafiği oluşturma
function createParticipationChart(participation) {
    const ctx = document.getElementById('participationChart').getContext('2d');
    
    // Daha önce oluşturulmuş bir grafik varsa yok et
    if (participationChart) {
        participationChart.destroy();
    }
    
    // Verileri hazırla
    const labels = Object.keys(participation);
    const data = Object.values(participation).map(value => value * 100); // Yüzdeye çevir
    
    // Rastgele renkler oluştur
    const colors = [
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 99, 132, 0.8)',
        'rgba(75, 192, 192, 0.8)',
        'rgba(255, 206, 86, 0.8)',
        'rgba(153, 102, 255, 0.8)'
    ];
    
    // Chart oluştur
    participationChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors.slice(0, data.length),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Video zamanda atlama
function jumpToVideoTime(seconds) {
    const videoPlayer = document.getElementById('videoPlayer');
    if (videoPlayer) {
        videoPlayer.currentTime = seconds;
        videoPlayer.play();
    }
}

// İşleme UI'ını sıfırla
function resetProcessingUI() {
    document.getElementById('uploadBtn').disabled = false;
    document.getElementById('uploadProgress').classList.add('d-none');
    document.querySelector('#uploadProgress .progress-bar').style.width = '0%';
    document.getElementById('processingSpinner').classList.add('d-none');
}

// Saniye cinsinden süreyi formatla (00:00 formatında)
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
} 