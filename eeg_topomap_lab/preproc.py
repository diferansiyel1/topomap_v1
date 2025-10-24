"""
Ön işleme modülü

Filtre, re-referencing, ICA/ASR artefakt giderme işlevleri.
"""

from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from rich.console import Console

console = Console()


class EEGPreprocessor:
    """EEG ön işleme sınıfı"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.ica = None
        self.ica_components = None
        
    def apply_filters(
        self,
        raw: mne.io.Raw,
        l_freq: Optional[float] = None,
        h_freq: Optional[float] = None,
        notch_freqs: Optional[Union[float, List[float]]] = None,
        notch_width: Optional[float] = None
    ) -> mne.io.Raw:
        """
        Filtre uygula
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        l_freq : float, optional
            Düşük frekans kesimi
        h_freq : float, optional
            Yüksek frekans kesimi
        notch_freqs : float or list, optional
            Notch filtre frekansları
        notch_width : float, optional
            Notch filtre genişliği
            
        Returns
        -------
        mne.io.Raw
            Filtrelenmiş veri
        """
        if self.verbose:
            console.print("[blue]Filtre uygulanıyor...[/blue]")
            
        # Band-pass filtre
        if l_freq is not None or h_freq is not None:
            raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
            if self.verbose:
                console.print(f"[green]Band-pass filtre: {l_freq}-{h_freq} Hz[/green]")
                
        # Notch filtre
        if notch_freqs is not None:
            if isinstance(notch_freqs, (int, float)):
                notch_freqs = [notch_freqs]
                
            for freq in notch_freqs:
                raw.notch_filter(freq, notch_widths=notch_width, verbose=False)
                if self.verbose:
                    console.print(f"[green]Notch filtre: {freq} Hz[/green]")
                    
        return raw
    
    def resample(self, raw: mne.io.Raw, sfreq: float) -> mne.io.Raw:
        """
        Yeniden örnekleme
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        sfreq : float
            Hedef örnekleme frekansı
            
        Returns
        -------
        mne.io.Raw
            Yeniden örneklenmiş veri
        """
        if raw.info['sfreq'] == sfreq:
            return raw
            
        if self.verbose:
            console.print(f"[blue]Yeniden örnekleme: {raw.info['sfreq']} -> {sfreq} Hz[/blue]")
            
        raw.resample(sfreq, verbose=False)
        return raw
    
    def apply_reference(
        self,
        raw: mne.io.Raw,
        ref_type: str = 'average',
        ref_channels: Optional[List[str]] = None
    ) -> mne.io.Raw:
        """
        Re-referencing uygula
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        ref_type : str
            Referans türü ('average', 'Cz', 'linked_ears', 'custom')
        ref_channels : list, optional
            Özel referans kanalları
            
        Returns
        -------
        mne.io.Raw
            Re-referenced veri
        """
        if self.verbose:
            console.print(f"[blue]Re-referencing: {ref_type}[/blue]")
            
        if ref_type == 'average':
            raw.set_eeg_reference(ref_channels='average', verbose=False)
        elif ref_type == 'Cz':
            if 'Cz' in raw.ch_names:
                raw.set_eeg_reference(ref_channels=['Cz'], verbose=False)
            else:
                console.print("[yellow]Cz kanalı bulunamadı, average referans kullanılıyor[/yellow]")
                raw.set_eeg_reference(ref_channels='average', verbose=False)
        elif ref_type == 'linked_ears':
            # Mastoid kanalları bul
            mastoid_chs = [ch for ch in raw.ch_names if 'M' in ch.upper() and ('1' in ch or '2' in ch)]
            if len(mastoid_chs) >= 2:
                raw.set_eeg_reference(ref_channels=mastoid_chs, verbose=False)
            else:
                console.print("[yellow]Mastoid kanalları bulunamadı, average referans kullanılıyor[/yellow]")
                raw.set_eeg_reference(ref_channels='average', verbose=False)
        elif ref_type == 'custom' and ref_channels:
            raw.set_eeg_reference(ref_channels=ref_channels, verbose=False)
        else:
            raise ValueError(f"Desteklenmeyen referans türü: {ref_type}")
            
        return raw
    
    def apply_ica(
        self,
        raw: mne.io.Raw,
        n_components: Optional[int] = None,
        method: str = 'fastica',
        random_state: int = 42,
        exclude_components: Optional[List[int]] = None,
        ecg_threshold: float = 0.1,
        eog_threshold: float = 0.1
    ) -> mne.io.Raw:
        """
        ICA artefakt giderme uygula
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        n_components : int, optional
            ICA bileşen sayısı
        method : str
            ICA metodu ('fastica', 'infomax', 'picard')
        random_state : int
            Rastgele tohum
        exclude_components : list, optional
            Hariç tutulacak bileşenler
        ecg_threshold : float
            EKG artefakt eşiği
        eog_threshold : float
            EOG artefakt eşiği
            
        Returns
        -------
        mne.io.Raw
            ICA uygulanmış veri
        """
        if self.verbose:
            console.print("[blue]ICA artefakt giderme uygulanıyor...[/blue]")
            
        # ICA nesnesi oluştur
        if n_components is None:
            n_components = min(64, len(raw.ch_names) - 1)
            
        self.ica = ICA(
            n_components=n_components,
            method=method,
            random_state=random_state,
            verbose=False
        )
        
        # ICA fit et
        self.ica.fit(raw, verbose=False)
        
        # Otomatik artefakt tespiti
        if exclude_components is None:
            exclude_components = []
            
        # EKG artefakt tespiti
        try:
            ecg_epochs = create_ecg_epochs(raw, ch_name='ECG' if 'ECG' in raw.ch_names else None)
            if ecg_epochs is not None:
                ecg_inds, ecg_scores = self.ica.find_bads_ecg(ecg_epochs, threshold=ecg_threshold)
                exclude_components.extend(ecg_inds)
                if self.verbose and ecg_inds:
                    console.print(f"[yellow]EKG artefakt bileşenleri: {ecg_inds}[/yellow]")
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]EKG artefakt tespiti başarısız: {e}[/yellow]")
                
        # EOG artefakt tespiti
        try:
            eog_epochs = create_eog_epochs(raw, ch_name='EOG' if 'EOG' in raw.ch_names else None)
            if eog_epochs is not None:
                eog_inds, eog_scores = self.ica.find_bads_eog(eog_epochs, threshold=eog_threshold)
                exclude_components.extend(eog_inds)
                if self.verbose and eog_inds:
                    console.print(f"[yellow]EOG artefakt bileşenleri: {eog_inds}[/yellow]")
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]EOG artefakt tespiti başarısız: {e}[/yellow]")
                
        # Bileşenleri hariç tut
        if exclude_components:
            self.ica.exclude = list(set(exclude_components))
            self.ica_components = self.ica.exclude
            
        # ICA uygula
        raw_ica = self.ica.apply(raw, exclude=self.ica.exclude, verbose=False)
        
        if self.verbose:
            console.print(f"[green]ICA uygulandı: {len(self.ica.exclude)} bileşen hariç tutuldu[/green]")
            
        return raw_ica
    
    def apply_asr(
        self,
        raw: mne.io.Raw,
        cutoff: float = 20.0,
        blocksize: int = 100,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25
    ) -> mne.io.Raw:
        """
        ASR (Artifact Subspace Reconstruction) uygula
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        cutoff : float
            ASR kesim değeri
        blocksize : int
            Blok boyutu
        max_dropout_fraction : float
            Maksimum dropout oranı
        min_clean_fraction : float
            Minimum temiz veri oranı
            
        Returns
        -------
        mne.io.Raw
            ASR uygulanmış veri
        """
        try:
            from mne.preprocessing import compute_current_source_density
            
            if self.verbose:
                console.print("[blue]ASR artefakt giderme uygulanıyor...[/blue]")
                
            # ASR için veriyi numpy array'e çevir
            data = raw.get_data()
            
            # Basit ASR implementasyonu (gerçek ASR için özel kütüphane gerekli)
            # Burada basit bir outlier detection uyguluyoruz
            clean_data = self._simple_asr(data, cutoff)
            
            # Yeni Raw objesi oluştur
            info = raw.info.copy()
            raw_asr = mne.io.RawArray(clean_data, info, verbose=False)
            
            if self.verbose:
                console.print("[green]ASR uygulandı[/green]")
                
            return raw_asr
            
        except ImportError:
            console.print("[yellow]ASR kütüphanesi bulunamadı, basit artefakt giderme uygulanıyor[/yellow]")
            return self._simple_artifact_removal(raw)
    
    def _simple_asr(self, data: np.ndarray, cutoff: float) -> np.ndarray:
        """Basit ASR implementasyonu"""
        # Z-score tabanlı outlier detection
        z_scores = np.abs((data - np.mean(data, axis=1, keepdims=True)) / 
                         np.std(data, axis=1, keepdims=True))
        
        # Outlier'ları ortalama ile değiştir
        outlier_mask = z_scores > cutoff
        clean_data = data.copy()
        clean_data[outlier_mask] = np.mean(data, axis=1, keepdims=True)[outlier_mask]
        
        return clean_data
    
    def _simple_artifact_removal(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Basit artefakt giderme"""
        data = raw.get_data()
        
        # Aşırı değerleri sınırla
        threshold = 3 * np.std(data)
        data = np.clip(data, -threshold, threshold)
        
        # Yeni Raw objesi oluştur
        info = raw.info.copy()
        raw_clean = mne.io.RawArray(data, info, verbose=False)
        
        return raw_clean
    
    def segment_data(
        self,
        raw: mne.io.Raw,
        segments: Dict[str, Tuple[float, float]],
        equal_length: bool = True
    ) -> Dict[str, mne.io.Raw]:
        """
        Veriyi segmentlere böl
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        segments : dict
            Segment adları ve zaman aralıkları
        equal_length : bool
            Tüm segmentleri eşit uzunlukta yap
            
        Returns
        -------
        dict
            Segment edilmiş veriler
        """
        if self.verbose:
            console.print("[blue]Veri segmentleniyor...[/blue]")
            
        segmented_data = {}
        
        # Eşit uzunluk için minimum uzunluğu bul
        if equal_length:
            min_length = min(end - start for start, end in segments.values())
            if self.verbose:
                console.print(f"[yellow]Eşit uzunluk: {min_length:.1f}s[/yellow]")
        
        for seg_name, (start, end) in segments.items():
            if equal_length:
                # Eşit uzunlukta segment
                actual_end = start + min_length
            else:
                actual_end = end
                
            # Segment oluştur
            seg_raw = raw.copy().crop(tmin=start, tmax=actual_end)
            segmented_data[seg_name] = seg_raw
            
            if self.verbose:
                console.print(f"[green]{seg_name}: {start:.1f}-{actual_end:.1f}s ({actual_end-start:.1f}s)[/green]")
                
        return segmented_data
    
    def get_preprocessing_info(self) -> Dict:
        """Ön işleme bilgilerini döndür"""
        info = {
            'ica_applied': self.ica is not None,
            'ica_components': self.ica_components if self.ica_components else [],
            'ica_method': self.ica.method if self.ica else None,
            'ica_n_components': self.ica.n_components_ if self.ica else None
        }
        return info


def create_preprocessing_pipeline(
    l_freq: Optional[float] = None,
    h_freq: Optional[float] = None,
    notch_freqs: Optional[Union[float, List[float]]] = None,
    ref_type: str = 'average',
    apply_ica: bool = False,
    apply_asr: bool = False,
    resample_freq: Optional[float] = None
) -> EEGPreprocessor:
    """
    Ön işleme pipeline'ı oluştur
    
    Parameters
    ----------
    l_freq : float, optional
        Düşük frekans kesimi
    h_freq : float, optional
        Yüksek frekans kesimi
    notch_freqs : float or list, optional
        Notch filtre frekansları
    ref_type : str
        Referans türü
    apply_ica : bool
        ICA uygula
    apply_asr : bool
        ASR uygula
    resample_freq : float, optional
        Yeniden örnekleme frekansı
        
    Returns
    -------
    EEGPreprocessor
        Ön işleme nesnesi
    """
    preprocessor = EEGPreprocessor()
    
    # Pipeline parametrelerini kaydet
    preprocessor.pipeline_params = {
        'l_freq': l_freq,
        'h_freq': h_freq,
        'notch_freqs': notch_freqs,
        'ref_type': ref_type,
        'apply_ica': apply_ica,
        'apply_asr': apply_asr,
        'resample_freq': resample_freq
    }
    
    return preprocessor


def apply_preprocessing_pipeline(
    raw: mne.io.Raw,
    preprocessor: EEGPreprocessor
) -> mne.io.Raw:
    """
    Ön işleme pipeline'ını uygula
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG verisi
    preprocessor : EEGPreprocessor
        Ön işleme nesnesi
        
    Returns
    -------
    mne.io.Raw
        Ön işlenmiş veri
    """
    params = getattr(preprocessor, 'pipeline_params', {})
    
    # Yeniden örnekleme
    if params.get('resample_freq'):
        raw = preprocessor.resample(raw, params['resample_freq'])
    
    # Filtre
    raw = preprocessor.apply_filters(
        raw,
        l_freq=params.get('l_freq'),
        h_freq=params.get('h_freq'),
        notch_freqs=params.get('notch_freqs')
    )
    
    # Re-referencing
    raw = preprocessor.apply_reference(raw, ref_type=params.get('ref_type', 'average'))
    
    # ICA
    if params.get('apply_ica'):
        raw = preprocessor.apply_ica(raw)
    
    # ASR
    if params.get('apply_asr'):
        raw = preprocessor.apply_asr(raw)
    
    return raw
