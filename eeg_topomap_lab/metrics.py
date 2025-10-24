"""
Metrik hesaplama modülü

Zaman domeni, frekans domeni ve nonlineer metrikler için kanal-bazlı hesaplamalar.
"""

from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from rich.console import Console

# MNE time_frequency modülünü import et
try:
    from mne.time_frequency import psd_array_welch
    # psd_welch fonksiyonunu psd_array_welch ile değiştir
    def psd_welch(raw, **kwargs):
        """psd_welch wrapper for psd_array_welch"""
        data = raw.get_data()
        freqs, psd = psd_array_welch(data, sfreq=raw.info['sfreq'], **kwargs)
        return psd, freqs
except ImportError:
    try:
        # MNE 1.10+ sürümleri için
        from mne import psd_welch
    except ImportError:
        # Eski MNE sürümleri için alternatif
        from mne.time_frequency import psd_welch as mne_psd_welch
        psd_welch = mne_psd_welch

# DFA ve entropy metrikleri için
try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False
    console = Console()
    console.print("[yellow]nolds kütüphanesi bulunamadı, DFA metrikleri kullanılamayacak[/yellow]")

try:
    import antropy as ant
    ANTROPY_AVAILABLE = True
except ImportError:
    ANTROPY_AVAILABLE = False
    console = Console()
    console.print("[yellow]antropy kütüphanesi bulunamadı, entropy metrikleri kullanılamayacak[/yellow]")

console = Console()


class EEGMetrics:
    """EEG metrik hesaplama sınıfı"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def calculate_time_domain_metrics(
        self,
        raw: mne.io.Raw,
        metrics: List[str] = ['rms', 'peak_to_peak', 'mean']
    ) -> Dict[str, np.ndarray]:
        """
        Zaman domeni metriklerini hesapla
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        metrics : list
            Hesaplanacak metrikler
            
        Returns
        -------
        dict
            Kanal-bazlı metrik değerleri
        """
        if self.verbose:
            console.print("[blue]Zaman domeni metrikleri hesaplanıyor...[/blue]")
            
        data = raw.get_data()
        results = {}
        
        for metric in metrics:
            if metric == 'rms':
                results[metric] = np.sqrt(np.mean(data**2, axis=1))
            elif metric == 'peak_to_peak':
                results[metric] = np.ptp(data, axis=1)
            elif metric == 'mean':
                results[metric] = np.mean(data, axis=1)
            elif metric == 'std':
                results[metric] = np.std(data, axis=1)
            elif metric == 'var':
                results[metric] = np.var(data, axis=1)
            else:
                console.print(f"[yellow]Bilinmeyen zaman domeni metriği: {metric}[/yellow]")
                
        return results
    
    def calculate_frequency_domain_metrics(
        self,
        raw: mne.io.Raw,
        metrics: List[str] = ['psd', 'band_power'],
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        relative: bool = False,
        welch_params: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Frekans domeni metriklerini hesapla
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        metrics : list
            Hesaplanacak metrikler
        bands : dict, optional
            Frekans bantları
        relative : bool
            Relatif güç hesapla
        welch_params : dict, optional
            Welch parametreleri
            
        Returns
        -------
        dict
            Kanal-bazlı metrik değerleri
        """
        if self.verbose:
            console.print("[blue]Frekans domeni metrikleri hesaplanıyor...[/blue]")
            
        # Varsayılan frekans bantları
        if bands is None:
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
            
        # Varsayılan Welch parametreleri
        if welch_params is None:
            welch_params = {
                'n_per_seg': 1024,
                'noverlap': 512,
                'nfft': 1024
            }
            
        results = {}
        
        # PSD hesapla
        if 'psd' in metrics:
            psd, freqs = mne.time_frequency.psd_welch(
                raw, 
                fmin=0.5, 
                fmax=100,
                **welch_params,
                verbose=False
            )
            results['psd'] = psd
            results['freqs'] = freqs
            
        # Bant güçleri
        if 'band_power' in metrics:
            band_powers = {}
            total_power = None
            
            for band_name, (fmin, fmax) in bands.items():
                band_psd, _ = psd_welch(
                    raw,
                    fmin=fmin,
                    fmax=fmax,
                    **welch_params,
                    verbose=False
                )
                
                # Bant gücü (integral)
                band_power = np.sum(band_psd, axis=1)
                band_powers[band_name] = band_power
                
                # Toplam güç (tüm bantlar için)
                if total_power is None:
                    total_psd, _ = psd_welch(
                        raw,
                        fmin=0.5,
                        fmax=100,
                        **welch_params,
                        verbose=False
                    )
                    total_power = np.sum(total_psd, axis=1)
                    
            # Relatif güç
            if relative:
                for band_name, band_power in band_powers.items():
                    rel_power = band_power / total_power
                    results[f'{band_name}_relative'] = rel_power
            else:
                results.update(band_powers)
                
        # Alfa/beta oranı
        if 'alpha_beta_ratio' in metrics and 'alpha' in bands and 'beta' in bands:
            alpha_power = band_powers.get('alpha', 0)
            beta_power = band_powers.get('beta', 0)
            ratio = alpha_power / (beta_power + 1e-10)  # Sıfıra bölme önleme
            results['alpha_beta_ratio'] = ratio
            
        return results
    
    def calculate_nonlinear_metrics(
        self,
        raw: mne.io.Raw,
        metrics: List[str] = ['dfa', 'lempel_ziv', 'higuchi_fd', 'permutation_entropy'],
        dfa_params: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Nonlineer metrikleri hesapla
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        metrics : list
            Hesaplanacak metrikler
        dfa_params : dict, optional
            DFA parametreleri
            
        Returns
        -------
        dict
            Kanal-bazlı metrik değerleri
        """
        if self.verbose:
            console.print("[blue]Nonlineer metrikler hesaplanıyor...[/blue]")
            
        data = raw.get_data()
        results = {}
        
        # Debug: data şeklini kontrol et
        if self.verbose:
            console.print(f"[blue]Data shape: {data.shape}[/blue]")
            console.print(f"[blue]Data type: {type(data)}[/blue]")
        
        # DFA parametreleri
        if dfa_params is None:
            dfa_params = {
                'n_windows': 20
            }
            
        for metric in metrics:
            if metric == 'dfa':
                if not NOLDS_AVAILABLE:
                    console.print("[yellow]DFA hesaplanamıyor: nolds kütüphanesi gerekli[/yellow]")
                    continue
                    
                dfa_values = []
                # data.shape = (n_channels, n_times)
                for ch_idx in range(data.shape[0]):
                    ch_data = data[ch_idx, :]  # Her kanalın zaman serisi
                    try:
                        # Veriyi numpy array'e çevir
                        ch_data = np.array(ch_data)
                        
                        # Debug: veri tipini kontrol et
                        if self.verbose:
                            console.print(f"[blue]Channel {ch_idx}: shape={ch_data.shape}, ndim={ch_data.ndim}, size={ch_data.size}[/blue]")
                        
                        # Veri tipini ve uzunluğunu kontrol et
                        if ch_data.ndim == 0 or ch_data.size == 0:
                            dfa_values.append(np.nan)
                            continue
                            
                        # 1D array'e çevir
                        if ch_data.ndim > 1:
                            ch_data = ch_data.flatten()
                            
                        if len(ch_data) < 100:  # DFA için minimum veri uzunluğu
                            dfa_values.append(np.nan)
                            continue
                            
                        # nolds.dfa için varsayılan parametreler
                        dfa_val = nolds.dfa(ch_data)
                        dfa_values.append(dfa_val)
                    except Exception as e:
                        console.print(f"[yellow]DFA hesaplama hatası: {e}[/yellow]")
                        dfa_values.append(np.nan)
                        
                results['dfa'] = np.array(dfa_values)
                
            elif metric == 'lempel_ziv':
                if not ANTROPY_AVAILABLE:
                    console.print("[yellow]Lempel-Ziv hesaplanamıyor: antropy kütüphanesi gerekli[/yellow]")
                    continue
                    
                lz_values = []
                for ch_idx in range(data.shape[0]):
                    ch_data = data[ch_idx, :]
                    try:
                        # Veriyi binary'ye çevir
                        binary_data = (ch_data > np.median(ch_data)).astype(int)
                        lz_val = ant.lziv_complexity(binary_data)
                        lz_values.append(lz_val)
                    except Exception as e:
                        console.print(f"[yellow]Lempel-Ziv hesaplama hatası: {e}[/yellow]")
                        lz_values.append(np.nan)
                        
                results['lempel_ziv'] = np.array(lz_values)
                
            elif metric == 'higuchi_fd':
                if not ANTROPY_AVAILABLE:
                    console.print("[yellow]Higuchi FD hesaplanamıyor: antropy kütüphanesi gerekli[/yellow]")
                    continue
                    
                hfd_values = []
                for ch_idx in range(data.shape[0]):
                    ch_data = data[ch_idx, :]
                    try:
                        hfd_val = ant.higuchi_fd(ch_data)
                        hfd_values.append(hfd_val)
                    except Exception as e:
                        console.print(f"[yellow]Higuchi FD hesaplama hatası: {e}[/yellow]")
                        hfd_values.append(np.nan)
                        
                results['higuchi_fd'] = np.array(hfd_values)
                
            elif metric == 'permutation_entropy':
                if not ANTROPY_AVAILABLE:
                    console.print("[yellow]Permutation Entropy hesaplanamıyor: antropy kütüphanesi gerekli[/yellow]")
                    continue
                    
                pe_values = []
                for ch_idx in range(data.shape[0]):
                    ch_data = data[ch_idx, :]
                    try:
                        pe_val = ant.perm_entropy(ch_data, order=3, normalize=True)
                        pe_values.append(pe_val)
                    except Exception as e:
                        console.print(f"[yellow]Permutation Entropy hesaplama hatası: {e}[/yellow]")
                        pe_values.append(np.nan)
                        
                results['permutation_entropy'] = np.array(pe_values)
                
            elif metric == 'sample_entropy':
                if not ANTROPY_AVAILABLE:
                    console.print("[yellow]Sample Entropy hesaplanamıyor: antropy kütüphanesi gerekli[/yellow]")
                    continue
                    
                se_values = []
                for ch_idx in range(data.shape[0]):
                    ch_data = data[ch_idx, :]
                    try:
                        # antropy.sample_entropy için doğru parametreler
                        se_val = ant.sample_entropy(ch_data, order=2, r=0.2*np.std(ch_data))
                        se_values.append(se_val)
                    except Exception as e:
                        console.print(f"[yellow]Sample Entropy hesaplama hatası: {e}[/yellow]")
                        se_values.append(np.nan)
                        
                results['sample_entropy'] = np.array(se_values)
                
        return results
    
    def calculate_all_metrics(
        self,
        raw: mne.io.Raw,
        time_metrics: List[str] = ['rms', 'peak_to_peak', 'mean'],
        freq_metrics: List[str] = ['band_power'],
        nonlinear_metrics: List[str] = ['dfa'],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Tüm metrikleri hesapla
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        time_metrics : list
            Zaman domeni metrikleri
        freq_metrics : list
            Frekans domeni metrikleri
        nonlinear_metrics : list
            Nonlineer metrikler
        **kwargs
            Diğer parametreler
            
        Returns
        -------
        dict
            Tüm metrik değerleri
        """
        all_results = {}
        
        # Zaman domeni
        if time_metrics:
            time_results = self.calculate_time_domain_metrics(raw, time_metrics)
            all_results.update(time_results)
            
        # Frekans domeni
        if freq_metrics:
            freq_results = self.calculate_frequency_domain_metrics(raw, freq_metrics, **kwargs)
            all_results.update(freq_results)
            
        # Nonlineer
        if nonlinear_metrics:
            nonlinear_results = self.calculate_nonlinear_metrics(raw, nonlinear_metrics, **kwargs)
            all_results.update(nonlinear_results)
            
        return all_results
    
    def get_channel_metrics_dataframe(
        self,
        metrics: Dict[str, np.ndarray],
        channel_names: List[str]
    ) -> pd.DataFrame:
        """
        Metrikleri DataFrame'e çevir
        
        Parameters
        ----------
        metrics : dict
            Metrik değerleri
        channel_names : list
            Kanal adları
            
        Returns
        -------
        pd.DataFrame
            Metrikler tablosu
        """
        df_data = {'channel': channel_names}
        
        for metric_name, values in metrics.items():
            if isinstance(values, np.ndarray) and values.ndim == 1:
                df_data[metric_name] = values
            elif isinstance(values, np.ndarray) and values.ndim == 2:
                # 2D array ise (örn. PSD), ortalama al
                df_data[metric_name] = np.mean(values, axis=1)
                
        return pd.DataFrame(df_data)


def calculate_dfa_detailed(
    signal: np.ndarray,
    min_window: int = 10,
    max_window: int = 100,
    n_windows: int = 20
) -> Tuple[float, Dict]:
    """
    Detaylı DFA hesaplama
    
    Parameters
    ----------
    signal : np.ndarray
        Sinyal verisi
    min_window : int
        Minimum pencere boyutu
    max_window : int
        Maksimum pencere boyutu
    n_windows : int
        Pencere sayısı
        
    Returns
    -------
    tuple
        DFA değeri ve detaylar
    """
    if not NOLDS_AVAILABLE:
        return np.nan, {}
        
    try:
        # DFA hesapla
        dfa_val = nolds.dfa(
            signal,
            nvals=n_windows,
            min_window=min_window,
            max_window=max_window
        )
        
        # Detaylı bilgi
        details = {
            'min_window': min_window,
            'max_window': max_window,
            'n_windows': n_windows,
            'signal_length': len(signal)
        }
        
        return dfa_val, details
        
    except Exception as e:
        console.print(f"[yellow]DFA hesaplama hatası: {e}[/yellow]")
        return np.nan, {}


def calculate_band_power_ratio(
    raw: mne.io.Raw,
    band1: Tuple[float, float],
    band2: Tuple[float, float],
    welch_params: Optional[Dict] = None
) -> np.ndarray:
    """
    İki bant arasındaki güç oranını hesapla
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG verisi
    band1 : tuple
        İlk bant (fmin, fmax)
    band2 : tuple
        İkinci bant (fmin, fmax)
    welch_params : dict, optional
        Welch parametreleri
        
    Returns
    -------
    np.ndarray
        Kanal-bazlı güç oranları
    """
    if welch_params is None:
        welch_params = {'n_per_seg': 1024, 'noverlap': 512}
        
    # Bant güçlerini hesapla
    band1_psd, _ = mne.time_frequency.psd_welch(
        raw, fmin=band1[0], fmax=band1[1], **welch_params, verbose=False
    )
    band2_psd, _ = mne.time_frequency.psd_welch(
        raw, fmin=band2[0], fmax=band2[1], **welch_params, verbose=False
    )
    
    # Güç oranı
    band1_power = np.sum(band1_psd, axis=1)
    band2_power = np.sum(band2_psd, axis=1)
    
    ratio = band1_power / (band2_power + 1e-10)
    
    return ratio


def calculate_spectral_edge_frequency(
    raw: mne.io.Raw,
    percentile: float = 95.0,
    fmin: float = 0.5,
    fmax: float = 100.0,
    welch_params: Optional[Dict] = None
) -> np.ndarray:
    """
    Spektral kenar frekansını hesapla
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG verisi
    percentile : float
        Yüzdelik dilim (örn. 95.0)
    fmin : float
        Minimum frekans
    fmax : float
        Maksimum frekans
    welch_params : dict, optional
        Welch parametreleri
        
    Returns
    -------
    np.ndarray
        Kanal-bazlı spektral kenar frekansları
    """
    if welch_params is None:
        welch_params = {'n_per_seg': 1024, 'noverlap': 512}
        
    # PSD hesapla
    psd, freqs = mne.time_frequency.psd_welch(
        raw, fmin=fmin, fmax=fmax, **welch_params, verbose=False
    )
    
    # Kümülatif güç
    cumsum_psd = np.cumsum(psd, axis=1)
    total_power = cumsum_psd[:, -1]
    
    # Yüzdelik dilim
    threshold = total_power * (percentile / 100.0)
    
    # Spektral kenar frekansı
    sef = np.zeros(psd.shape[0])
    for i in range(psd.shape[0]):
        idx = np.where(cumsum_psd[i, :] >= threshold[i])[0]
        if len(idx) > 0:
            sef[i] = freqs[idx[0]]
        else:
            sef[i] = freqs[-1]
            
    return sef
