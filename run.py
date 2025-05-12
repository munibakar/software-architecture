import torch
import os
from model.app import app

if __name__ == '__main__':
    # GPU kullanımını optimize et
    if torch.cuda.is_available():
        print(f"GPU Bulundu: {torch.cuda.get_device_name(0)}")
        print(f"GPU Sayısı: {torch.cuda.device_count()}")
        
        # CUDA bellek yönetimi için ayarlar
        torch.cuda.empty_cache()
        
        # Otomatik karışık hassasiyet (mixed precision) kullanımını etkinleştir
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Veri tipi tutarlılığı için varsayılan tipi float32 olarak ayarla
        # (float16 bazı modellerde uyumsuzluk sorunlarına neden oluyor)
        torch.set_default_dtype(torch.float32)
        
        # CUDA bellek yönetimini iyileştir
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("GPU bulunamadı, CPU kullanılacak.")
        # CPU için varsayılan tipi ayarla
        torch.set_default_dtype(torch.float32)

    # Flask uygulamasını başlat
    app.run(host='0.0.0.0', port=5000, debug=False) 