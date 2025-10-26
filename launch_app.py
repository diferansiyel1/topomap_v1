#!/usr/bin/env python3
"""
EEG Topomap Lab - Ultra Güvenli Başlatıcı
Bu script uygulamayı ultra güvenli bir şekilde başlatır ve tarayıcıda açar.
"""

import subprocess
import sys
import os
import time
import webbrowser
import socket
import signal
from pathlib import Path

def is_port_available(port):
    """Port'un boş olup olmadığını kontrol et"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_app_path():
    """PyInstaller ve normal Python için doğru app.py path'ini bul"""
    # PyInstaller için
    if hasattr(sys, '_MEIPASS'):
        app_path = Path(sys._MEIPASS) / "eeg_topomap_lab" / "app.py"
        if app_path.exists():
            return app_path
    
    # Normal Python için
    script_dir = Path(__file__).parent
    app_path = script_dir / "eeg_topomap_lab" / "app.py"
    if app_path.exists():
        return app_path
    
    # Alternatif yollar
    for alt_path in [
        Path.cwd() / "eeg_topomap_lab" / "app.py",
        Path(__file__).parent.parent / "eeg_topomap_lab" / "app.py"
    ]:
        if alt_path.exists():
            return alt_path
    
    return None

def check_streamlit_running(port):
    """Streamlit'in gerçekten çalışıp çalışmadığını kontrol et"""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Ana başlatma fonksiyonu - Ultra güvenli versiyon"""
    print("🧠 EEG Topomap Lab Ultra Güvenli Başlatıcı")
    print("=" * 60)
    
    # Önce mevcut Streamlit process'lerini temizle
    print("🧹 Mevcut Streamlit process'leri temizleniyor...")
    try:
        subprocess.run(["pkill", "-f", "streamlit"], check=False)
        time.sleep(2)
    except:
        pass
    
    # App.py dosyasını bul
    app_path = find_app_path()
    if not app_path:
        print("❌ KRİTİK HATA: app.py dosyası bulunamadı!")
        print("📁 Kontrol edilen yerler:")
        if hasattr(sys, '_MEIPASS'):
            print(f"   - PyInstaller: {sys._MEIPASS}/eeg_topomap_lab/app.py")
        print(f"   - Script dizini: {Path(__file__).parent}/eeg_topomap_lab/app.py")
        print(f"   - Mevcut dizin: {Path.cwd()}/eeg_topomap_lab/app.py")
        print("\n🛑 Güvenlik nedeniyle uygulama başlatılmadı!")
        input("Devam etmek için Enter'a basın...")
        return 1
    
    print(f"✅ app.py bulundu: {app_path}")
    
    # Port kontrolü
    port = 8501
    if not is_port_available(port):
        print(f"⚠️  Port {port} dolu! Alternatif port aranıyor...")
        for alt_port in range(8502, 8520):
            if is_port_available(alt_port):
                port = alt_port
                print(f"✅ Port {port} kullanılacak")
                break
        else:
            print("❌ Uygun port bulunamadı! Lütfen diğer Streamlit uygulamalarını kapatın.")
            input("Devam etmek için Enter'a basın...")
            return 1
    
    # Streamlit komutunu hazırla
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.runOnSave", "false",
        "--server.enableCORS", "false"
    ]
    
    print(f"🚀 Port {port}'da başlatılıyor...")
    
    try:
        # Streamlit'i başlat
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Process'in başarıyla başladığını kontrol et
        print("⏳ Streamlit başlatılıyor...")
        time.sleep(5)
        
        # Process hala çalışıyor mu kontrol et
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("❌ Streamlit başlatılamadı!")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return 1
        
        # Streamlit'in gerçekten çalışıp çalışmadığını kontrol et
        print("🔍 Streamlit durumu kontrol ediliyor...")
        time.sleep(3)
        
        if check_streamlit_running(port):
            print(f"✅ Streamlit başarıyla çalışıyor!")
            
            # Güvenli tarayıcı açma
            url = f"http://localhost:{port}"
            print(f"🌐 Tarayıcıda açılıyor: {url}")
            webbrowser.open(url)
            
            print("✅ EEG Topomap Lab başarıyla başlatıldı!")
            print("📊 Uygulamayı kapatmak için Ctrl+C tuşlarına basın.")
            
            # Process'i bekle
            process.wait()
        else:
            print("❌ Streamlit çalışmıyor! Process sonlandırılıyor.")
            process.terminate()
            return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Uygulama kapatılıyor...")
        if 'process' in locals():
            process.terminate()
        return 0
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
        if 'process' in locals():
            process.terminate()
        return 1

if __name__ == "__main__":
    sys.exit(main())

if __name__ == "__main__":
    sys.exit(main())

