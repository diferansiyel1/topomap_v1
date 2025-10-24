"""
Streamlit GUI arayüzü

Web tabanlı kullanıcı arayüzü.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from . import bipolar
except ImportError:
    # Streamlit doğrudan çalıştırıldığında absolute import kullan
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from eeg_topomap_lab import bipolar

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="EEG Topomap Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Ana Streamlit uygulaması"""
    
    # Başlık
    st.markdown('<h1 class="main-header">🧠 EEG Topomap Lab</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Ana içerik - sadece görselleştirme sekmesi
    st.info("📊 Bu uygulama, hesaplanmış EEG metrik verilerinizi (DFA, PSD vb.) CSV formatında yapıştırarak topolojik haritalar oluşturmanızı sağlar.")
    
    # Tab'lar - sadece görselleştirme
    tab1 = st.tabs(["📈 EEG Metrik Görselleştirme"])[0]
    
    with tab1:
        visualization_tab()




def visualization_tab():
    """Görselleştirme sekmesi"""
    st.header("📈 EEG Metrik Verileri Görselleştirme")
    
    # EEG Metrik Verileri Girişi
    st.subheader("📊 EEG Metrik Verileri Girişi")
    st.info("Hesaplanmış EEG metrik verilerinizi (DFA, PSD vb.) CSV formatında yapıştırabilirsiniz. Bipolar veya unipolar kanallar desteklenir.")
    st.info("💡 **Örnek Format:** CSV verinizin ilk sütunu kanal adları, diğer sütunlar farklı koşulları (örn. Erken-Pre, Geç-Pre) temsil etmelidir.")
    
    # Veri yapıştırma alanı
    st.markdown("**Veri Formatı Seçenekleri:**")
    data_format = st.radio(
        "Veri formatını seçin:",
        ["CSV Format", "Doğrudan Kopyala-Yapıştır (Tab/Boşluk ile ayrılmış)"],
        index=1
    )
    
    if data_format == "CSV Format":
        metric_data_input = st.text_area(
            "Metrik Verileri Yapıştırın (CSV formatında):",
            height=200,
            help="Örnek format:\nKanal,Erken-Pre,Geç-Pre\nFP1-F7,0.80395722,0.64169454\nF7-T7,0.77854514,0.55379151\n\nVeya:\nKanal,DFA\nF3-C3,0.53246278\nF4-C4,0.53986818"
        )
    else:
        st.info("💡 **Doğrudan Kopyala-Yapıştır:** Verilerinizi tab veya boşluk ile ayrılmış olarak yapıştırabilirsiniz. Örnek:")
        st.code("""Predicted means (LS means)	Erken - Pre	Geç - Pre
FP1-F7	1,784	2,117
F7-T7	3,773	3,267
T7-P7	6,114	4,117""")
        
        metric_data_input = st.text_area(
            "Verileri Yapıştırın (Tab/Boşluk ile ayrılmış):",
            height=300,
            help="Verilerinizi doğrudan kopyalayıp buraya yapıştırabilirsiniz. Tab veya boşluk ile ayrılmış olmalıdır."
        )
    
    if metric_data_input:
        try:
            import io
            import pandas as pd
            import matplotlib.pyplot as plt
            import mne
            import numpy as np
            
            # Veri formatına göre okuma
            if data_format == "CSV Format":
                # CSV verisini oku - virgül ve noktalı virgül desteği
                try:
                    df_metrics = pd.read_csv(io.StringIO(metric_data_input), sep=',')
                except:
                    # Türkçe format için noktalı virgül desteği
                    df_metrics = pd.read_csv(io.StringIO(metric_data_input), sep=';')
                    # Virgülleri nokta ile değiştir (Türkçe sayı formatı)
                    for col in df_metrics.columns:
                        if df_metrics[col].dtype == 'object':
                            try:
                                df_metrics[col] = df_metrics[col].astype(str).str.replace(',', '.').astype(float)
                            except:
                                pass  # Kanal adları için hata verme
            else:
                # Tab/boşluk ile ayrılmış veri okuma
                try:
                    # Önce tab ile deneyelim
                    df_metrics = pd.read_csv(io.StringIO(metric_data_input), sep='\t')
                except:
                    try:
                        # Sonra boşluk ile deneyelim
                        df_metrics = pd.read_csv(io.StringIO(metric_data_input), sep='\s+', engine='python')
                    except:
                        # Son çare olarak virgül ile deneyelim
                        df_metrics = pd.read_csv(io.StringIO(metric_data_input), sep=',')
                
                # Türkçe sayı formatını düzelt (virgül -> nokta)
                for col in df_metrics.columns:
                    if df_metrics[col].dtype == 'object':
                        try:
                            # Virgülleri nokta ile değiştir
                            df_metrics[col] = df_metrics[col].astype(str).str.replace(',', '.')
                            # Sayısal sütunları float'a çevir
                            df_metrics[col] = pd.to_numeric(df_metrics[col], errors='ignore')
                        except:
                            pass  # Kanal adları için hata verme
            
            st.success(f"✅ {len(df_metrics)} kanal verisi yüklendi")
            
            # Veriyi göster
            st.dataframe(df_metrics, width='stretch')
            
            # Sütun isimlerini kontrol et ve kanal sütununu bul
            possible_channel_cols = ['Kanal', 'Channel', 'kanal', 'channel', 'Ch', 'ch']
            channel_col = None
            for col in possible_channel_cols:
                if col in df_metrics.columns:
                    channel_col = col
                    break
            
            if channel_col is None and len(df_metrics.columns) > 0:
                # İlk sütunu kanal olarak kabul et
                channel_col = df_metrics.columns[0]
                st.warning(f"Kanal sütunu bulunamadı, '{channel_col}' kullanılıyor")
            
            # Metrik sütunlarını belirle (kanal sütunu dışındakiler)
            metric_cols = [col for col in df_metrics.columns if col != channel_col and not col.startswith('Unnamed')]
            
            if len(metric_cols) == 0:
                st.error("❌ Metrik sütunu bulunamadı!")
                return
            
            # Bipolar işleyici oluştur
            bipolar_processor = bipolar.BipolarProcessor(verbose=True)
            
            # Topomap parametreleri
            col1, col2 = st.columns(2)
            with col1:
                montage_type = st.selectbox("Montaj", ['standard_1020', 'standard_1005'], index=0)
                cmap = st.selectbox("Renk Haritası", ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
            with col2:
                show_contours = st.checkbox("Konturları göster", value=True)
                show_names = st.checkbox("Kanal adlarını göster", value=True)
                contours_count = st.slider("Kontur sayısı", 0, 20, 6) if show_contours else 0
            
            # Ortak ölçekleme için tüm değerleri topla
            all_values = []
            for metric_col in metric_cols:
                channel_data = dict(zip(df_metrics[channel_col], df_metrics[metric_col]))
                values, _, _ = bipolar_processor.create_bipolar_topomap_data(
                    channel_data, montage=montage_type
                )
                if len(values) > 0:
                    all_values.extend(values)
            
            # Ortak vmin ve vmax hesapla
            if all_values:
                common_vmin = min(all_values)
                common_vmax = max(all_values)
                st.info(f"📊 Ortak ölçekleme: {common_vmin:.3f} - {common_vmax:.3f}")
            else:
                common_vmin = None
                common_vmax = None
            
            # Topomap oluştur butonu
            if st.button("🗺️ Topolojik Haritalar Oluştur", type="primary"):
                try:
                    # Her metrik sütunu için topomap oluştur
                    for metric_col in metric_cols:
                        st.subheader(f"📈 {metric_col}")
                        
                        # Kanal verilerini hazırla
                        channel_data = dict(zip(df_metrics[channel_col], df_metrics[metric_col]))
                        
                        # Bipolar topomap verilerini oluştur
                        values, channel_names, positions = bipolar_processor.create_bipolar_topomap_data(
                            channel_data, montage=montage_type
                        )
                        
                        # Debug bilgileri göster
                        st.write(f"**Debug Bilgileri - {metric_col}:**")
                        st.write(f"- Toplam kanal sayısı: {len(channel_data)}")
                        st.write(f"- Koordinat bulunan kanal sayısı: {len(values)}")
                        st.write(f"- Bulunan kanallar: {channel_names}")
                        st.write(f"- Koordinat bulunamayan kanallar: {set(channel_data.keys()) - set(channel_names)}")
                        
                        if len(values) > 0:
                            # MNE Info objesi oluştur
                            info = mne.create_info(channel_names, sfreq=250, ch_types='eeg')
                            
                            # Özel koordinatları ayarla
                            if positions:
                                # MNE DigMontage oluştur
                                from mne.channels import make_dig_montage
                                custom_montage = make_dig_montage(ch_pos=positions, coord_frame='head')
                                info.set_montage(custom_montage)
                            
                            # Topomap çiz
                            from eeg_topomap_lab.viz import EEGVisualizer
                            visualizer = EEGVisualizer(verbose=False)
                            
                            try:
                                fig = visualizer.plot_topomap(
                                    values=values,
                                    info=info,
                                    vmin=common_vmin,
                                    vmax=common_vmax,
                                    cmap=cmap,
                                    contours=contours_count,
                                    show_names=show_names,
                                    title=f"{metric_col} - Topolojik Harita"
                                )
                                
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # İstatistikler
                                col_stats1, col_stats2, col_stats3 = st.columns(3)
                                with col_stats1:
                                    st.metric("Kanal Sayısı", len(values))
                                with col_stats2:
                                    st.metric("Min Değer", f"{values.min():.6f}")
                                with col_stats3:
                                    st.metric("Max Değer", f"{values.max():.6f}")
                                    
                            except Exception as e:
                                st.error(f"❌ {metric_col} topomap çizim hatası: {e}")
                                
                        else:
                            st.warning(f"⚠️ {metric_col} için geçerli koordinat bulunamadı")
                    
                    # Çoklu panel görünümü (birden fazla koşul varsa)
                    if len(metric_cols) > 1:
                        st.subheader("📊 Çoklu Panel Görünümü")
                        
                        if st.button("🖼️ Çoklu Panel Oluştur"):
                            try:
                                # Tüm koşulları bir arada göster
                                data_dict = {}
                                for metric_col in metric_cols:
                                    channel_data = dict(zip(df_metrics[channel_col], df_metrics[metric_col]))
                                    values, channel_names, positions = bipolar_processor.create_bipolar_topomap_data(
                                        channel_data, montage=montage_type
                                    )
                                    if len(values) > 0 and len(channel_names) > 0:
                                        data_dict[metric_col] = values
                                
                                if len(data_dict) > 0:
                                    # İlk koşulun info objesini kullan
                                    first_col = metric_cols[0]
                                    channel_data = dict(zip(df_metrics[channel_col], df_metrics[first_col]))
                                    _, channel_names, positions = bipolar_processor.create_bipolar_topomap_data(
                                        channel_data, montage=montage_type
                                    )
                                    
                                    if positions:
                                        info = mne.create_info(channel_names, sfreq=250, ch_types='eeg')
                                        from mne.channels import make_dig_montage
                                        custom_montage = make_dig_montage(ch_pos=positions, coord_frame='head')
                                        info.set_montage(custom_montage)
                                        
                                        # Çoklu panel topomap
                                        rows = (len(data_dict) + 1) // 2
                                        cols = min(2, len(data_dict))
                                        fig = visualizer.plot_multi_panel_topomap(
                                            data_dict=data_dict,
                                            info=info,
                                            rows=rows,
                                            cols=cols,
                                            vmin=common_vmin,  # Ortak ölçekleme
                                            vmax=common_vmax,
                                            cmap=cmap,
                                            contours=contours_count,
                                            show_names=show_names
                                        )
                                        
                                        st.pyplot(fig)
                                        plt.close(fig)
                            except Exception as e:
                                st.error(f"❌ Çoklu panel hatası: {e}")
                                
                except Exception as e:
                    st.error(f"❌ Topomap oluşturma hatası: {e}")
                    import traceback
                    st.error(f"Detay: {traceback.format_exc()}")
                    
        except Exception as e:
            st.error(f"❌ Veri okuma hatası: {e}")
            import traceback
            st.error(f"Detay: {traceback.format_exc()}")
    



if __name__ == "__main__":
    main()
