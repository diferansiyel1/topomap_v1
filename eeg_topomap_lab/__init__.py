"""
EEG Topomap Lab - EEG verilerinden topolojik harita analizi

Bu paket EEG verilerinden yayın kalitesinde topolojik haritalar üretmek için
kapsamlı araçlar sağlar. Zaman domeni, frekans domeni ve nonlineer metrikler
için kanal-bazlı değerleri hesaplar ve topomap oluşturur.

Ana modüller:
- io: Veri giriş/çıkış işlemleri
- preproc: Ön işleme (filtre, re-referencing, ICA)
- metrics: Metrik hesaplamaları (PSD, DFA, nonlineer)
- bipolar: Bipolar veri işleme
- stats: İstatistiksel analiz
- viz: Görselleştirme
- export: Dışa aktarma
- cli: Komut satırı arayüzü
- app: Streamlit GUI
"""

__version__ = "0.1.0"
__author__ = "EEG Topomap Lab"
__email__ = "info@eegtopomap.dev"

from . import io
from . import preproc
from . import metrics
from . import bipolar
from . import stats
from . import viz
from . import export

__all__ = [
    "io",
    "preproc", 
    "metrics",
    "bipolar",
    "stats",
    "viz",
    "export",
]
