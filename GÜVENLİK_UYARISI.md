# 🚨 KRİTİK GÜVENLİK UYARISI

## Problem Özeti

PyInstaller ile paketlenen executable'lar (`EEG_Topomap_Lab_UltraSafe.app`), process kontrolünü atlamaktadır. Bu nedenle:

1. ✅ Executable **1 KERE** tıklandığında düzgün çalışır
2. ❌ Executable **TEKRAR TEKRAR** tıklandığında yüzlerce process oluşturur
3. 🔥 Bu, bilgisayarın ısınmasına ve çökmesine neden olabilir

## 🔍 Problem Nedeni

PyInstaller ile paketlenen executable'larda:
- Process kontrolü çalışmıyor
- Streamlit'in başarıyla çalışıp çalışmadığını kontrol edemiyoruz
- Tarayıcı açma işlemi sonsuz döngüye girebiliyor

## ✅ Güvenli Çözüm

**ÖNERİLEN YÖNTEM**: Executable kullanmak yerine, Python script'i manuel olarak çalıştırın:

```bash
# Terminal'den:
python launch_app.py
```

Bu yöntem:
- ✅ Process kontrolü çalışır
- ✅ Sonsuz döngü riski yok
- ✅ Güvenli ve stabil

## 🎯 Executable Kullanım Kuralları

Eğer yine de executable'ı kullanmak istiyorsanız:

1. **SADECE 1 KERE tıklayın**
2. Bekleyin (10-15 saniye)
3. Eğer açılmazsa, executable'ı kapatıp **tekrar 1 kere** çalıştırın
4. **ASLA** tekrar tekrar tıklamayın!

## 🛠️ Teknik Detaylar

- Python script'i (`launch_app.py`): ✅ Güvenli
- PyInstaller executable: ⚠️ Sadece 1 kere tıklanmalı
- Process kontrolü: Sadece Python script'te çalışıyor

## 📝 Not

Bu problem PyInstaller'ın bir kısıtlamasıdır. Executable güvenli hale getirmek için:
1. Process kontrolünü PyInstaller'a eklemek gerekir (karmaşık)
2. Manuel başlatma kullanmak (ÖNERİLEN)

**ÖNERİM**: Executable yerine Python script'i kullanın!

