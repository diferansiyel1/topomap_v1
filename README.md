# EEG Topomap Lab

EEG verilerinden yayın kalitesinde topolojik haritalar üretmek için kapsamlı Python paketi.

## Özellikler

- **Çoklu Format Desteği**: EDF, BDF, FIF, CSV dosya formatları
- **Bipolar/Unipolar Veri**: Bipolar verilerden referans dönüşümü veya orta-nokta yaklaşımı
- **Kapsamlı Metrikler**: Zaman domeni, frekans domeni ve nonlineer metrikler (DFA, Lempel-Ziv, Higuchi FD)
- **İstatistiksel Analiz**: Tek grup, iki grup ve çoklu karşılaştırma testleri
- **Yayın Kalitesi**: SVG/PNG/PDF çıktıları, ortak renk skalası, istatistiksel maskeleme
- **CLI ve GUI**: Komut satırı arayüzü ve Streamlit tabanlı web arayüzü

## Kurulum

```bash
# Geliştirme ortamı için
git clone <repository>
cd topomap
pip install -e ".[dev]"

# Sadece kullanım için
pip install eeg-topomap-lab
```

## Hızlı Başlangıç

### CLI Kullanımı

```bash
# Bipolar veri → referans dönüşüm + DFA topomap
eegtopo \
  --input data/subject01.edf \
  --montage standard_1020 \
  --bipolar-to-ref average \
  --segments "preiktal:300-600" "interiktal:1200-1500" \
  --metric dfa --dfa-min 10 --dfa-max 100 \
  --band alpha --psd-welch nperseg=1024 \
  --compare preiktal interiktal --paired false \
  --fdr 0.05 \
  --topo vmin=0.5 vmax=1.2 cmap=viridis contours=0 show_names=false \
  --export fig/out_pre_vs_inter_alpha_dfa.svg \
  --meta fig/out_pre_vs_inter_alpha_dfa.json
```

### GUI Kullanımı

```bash
streamlit run eeg_topomap_lab/app.py
```

## Desteklenen Metrikler

### Zaman Domeni
- RMS (Root Mean Square)
- Tepe-tepe değeri
- ERP ortalaması

### Frekans Domeni
- PSD (Power Spectral Density)
- Bant güçleri (delta, theta, alpha, beta, gamma)
- Relatif güç
- Alfa/beta oranı

### Nonlineer/Kompleksite
- DFA (Detrended Fluctuation Analysis)
- Lempel-Ziv kompleksitesi
- Higuchi Fractal Dimension
- Permutation Entropy

## Konfigürasyon

YAML dosyası ile tüm parametreleri yapılandırabilirsiniz:

```yaml
# configs/example.yaml
input:
  file: "data/subject01.edf"
  montage: "standard_1020"
  bipolar_to_ref: "average"

segments:
  preiktal: [300, 600]
  interiktal: [1200, 1500]

metrics:
  type: "dfa"
  dfa_min: 10
  dfa_max: 100

statistics:
  compare: ["preiktal", "interiktal"]
  paired: false
  fdr: 0.05

visualization:
  vmin: 0.5
  vmax: 1.2
  cmap: "viridis"
  contours: 0
  show_names: false

export:
  figure: "fig/out_pre_vs_inter_alpha_dfa.svg"
  metadata: "fig/out_pre_vs_inter_alpha_dfa.json"
```

```bash
eegtopo --config configs/example.yaml
```

## Test

```bash
pytest tests/ -v
```

## Lisans

MIT License

## Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## Bilimsel Atıflar

Bu paket aşağıdaki kütüphaneleri kullanır:
- MNE-Python (Gramfort et al., 2013)
- nolds (DFA implementasyonu)
- antropy (entropy metrikleri)
- pingouin (istatistiksel testler)
