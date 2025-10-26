# 🧠 EEG Topomap Lab - Tek Tıkla Başlatıcı

## ⚠️ ÖNEMLİ GÜVENLİK UYARISI

**PyInstaller ile paketlenen executable'larda process kontrolü çalışmıyor.** Bu nedenle executable'ı çalıştırdığınızda **tekrar tekrar tıklamayın**. Eğer uygulama açılmıyorsa, executable'ı **SADECE 1 KERE** çalıştırın ve bekleyin.

## 📋 Açıklama

EEG Topomap Lab, bipolar EEG verilerini analiz edip topolojik haritalar oluşturan profesyonel bir uygulamadır. Bu paket, uygulamayı tek tıkla başlatmanızı sağlar.

## 🚀 Nasıl Kullanılır

### ÖNERİLEN YÖNTEM: Manuel Terminal Başlatma

```bash
# Terminal'den başlatmak için (ÖNERİLEN):
python launch_app.py
```

### macOS Executable (.app):

1. **EEG_Topomap_Lab_UltraSafe.app** dosyasını çift tıklayın
2. **SADECE 1 KERE** tıklayın ve bekleyin
3. Uygulama otomatik olarak başlayacak ve tarayıcınızda açılacak
4. EEG verilerinizi yükleyip analiz edebilirsiniz

### Manuel Başlatma (Alternatif):

```bash
# Terminal'den executable'ı başlatmak için:
open "dist/EEG_Topomap_Lab_UltraSafe.app"
```

## 📊 Özellikler

- ✅ **Bipolar EEG Analizi**: 22 bipolar kanal desteği
- ✅ **Otomatik Koordinat Hesaplama**: MNE standard 1020 montajı
- ✅ **Yuvarlak Kafa Şekli**: Klasik EEG topomap görünümü
- ✅ **Yüksek Çözünürlük**: 128x128 piksel kalitesi
- ✅ **Streamlit Arayüzü**: Modern ve kullanıcı dostu
- ✅ **Tek Tıkla Başlatma**: Hiçbir kurulum gerektirmez

## 🔧 Teknik Detaylar

- **Platform**: macOS (ARM64)
- **Python**: 3.13.3
- **Framework**: Streamlit + MNE-Python
- **Paketleme**: PyInstaller
- **Boyut**: ~1.5 MB executable

## 📁 Dosya Yapısı

```
dist/
├── EEG_Topomap_Lab.app/          # macOS uygulama paketi
│   ├── Contents/
│   │   ├── MacOS/
│   │   │   └── EEG_Topomap_Lab   # Ana executable
│   │   └── Resources/
│   └── ...
└── EEG_Topomap_Lab/              # Alternatif dizin paketi
    ├── EEG_Topomap_Lab           # Executable
    ├── eeg_topomap_lab/          # Uygulama modülleri
    └── ...
```

## 🎯 Desteklenen EEG Formatları

- CSV dosyaları (tab-separated)
- Bipolar kanal formatı (örn: FP1-F7, F7-T7)
- Pre-Ictal ve Inter-Ictal veri karşılaştırması

## 🛠️ Sorun Giderme

### Uygulama açılmıyorsa:

1. **macOS Güvenlik**: Sistem Tercihleri > Güvenlik > Genel'den "İzin Ver" seçin
2. **Port Çakışması**: Başka bir Streamlit uygulaması çalışıyorsa kapatın
3. **Tarayıcı**: Chrome, Safari veya Firefox kullanın

### Manuel Başlatma:

```bash
# Terminal'den:
cd "dist/EEG_Topomap_Lab"
./EEG_Topomap_Lab
```

## 📞 Destek

Herhangi bir sorun yaşarsanız:
- Terminal çıktısını kontrol edin
- Port 8501'in boş olduğundan emin olun
- macOS sürümünüzün uyumlu olduğunu kontrol edin

## 🎉 Başarılı Kurulum!

Artık EEG Topomap Lab'ı tek tıkla başlatabilirsiniz! 

**Not**: İlk açılışta macOS güvenlik uyarısı çıkabilir. "İzin Ver" butonuna tıklayarak devam edin.

