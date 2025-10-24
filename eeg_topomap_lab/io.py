"""
Veri giriş/çıkış modülü

EDF, BDF, FIF, CSV formatlarını destekler.
Kanal adlandırma düzeltme ve montaj yükleme işlevleri.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from mne.io import read_raw_edf, read_raw_bdf, read_raw_fif
from rich.console import Console

console = Console()


class EEGDataLoader:
    """EEG veri yükleme ve işleme sınıfı"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.raw = None
        self.montage = None
        self.channel_mapping = {}
        
    def load_data(
        self, 
        file_path: Union[str, Path],
        montage: str = "standard_1020",
        channel_mapping: Optional[Dict[str, str]] = None,
        preload: bool = True
    ) -> mne.io.Raw:
        """
        EEG verisini yükle ve montaj uygula
        
        Parameters
        ----------
        file_path : str or Path
            EEG dosya yolu
        montage : str
            Montaj adı ('standard_1020', 'standard_1005') veya dosya yolu
        channel_mapping : dict, optional
            Kanal adı eşleme sözlüğü
        preload : bool
            Veriyi belleğe yükle
            
        Returns
        -------
        mne.io.Raw
            Yüklenen EEG verisi
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
            
        # Dosya formatını belirle
        file_ext = file_path.suffix.lower()
        
        if self.verbose:
            console.print(f"[green]Dosya yükleniyor: {file_path}[/green]")
            
        try:
            # Dosya formatına göre yükle
            if file_ext == '.edf':
                self.raw = read_raw_edf(file_path, preload=preload, verbose=False)
            elif file_ext == '.bdf':
                self.raw = read_raw_bdf(file_path, preload=preload, verbose=False)
            elif file_ext in ['.fif', '.fif.gz']:
                self.raw = read_raw_fif(file_path, preload=preload, verbose=False)
            elif file_ext == '.csv':
                self.raw = self._load_csv(file_path)
            else:
                raise ValueError(f"Desteklenmeyen dosya formatı: {file_ext}")
                
        except Exception as e:
            raise RuntimeError(f"Dosya yükleme hatası: {e}")
            
        # Kanal adlarını düzelt
        if channel_mapping:
            self.channel_mapping = channel_mapping
            self._fix_channel_names()
            
        # Montaj uygula
        self._apply_montage(montage)
        
        if self.verbose:
            console.print(f"[green]Veri yüklendi: {len(self.raw.ch_names)} kanal, {self.raw.times[-1]:.1f}s[/green]")
            
        return self.raw
    
    def _load_csv(self, file_path: Path) -> mne.io.Raw:
        """CSV dosyasından EEG verisi yükle"""
        try:
            # CSV'yi oku
            df = pd.read_csv(file_path)
            
            # Kanal adlarını al (ilk sütun zaman olabilir)
            if 'time' in df.columns.str.lower():
                ch_names = df.columns[1:].tolist()
                data = df.iloc[:, 1:].values.T
            else:
                ch_names = df.columns.tolist()
                data = df.values.T
                
            # Örnekleme frekansını tahmin et (varsayılan 250 Hz)
            sfreq = 250.0
            
            # MNE RawArray oluştur
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=sfreq,
                ch_types='eeg'
            )
            
            raw = mne.io.RawArray(data, info, verbose=False)
            return raw
            
        except Exception as e:
            raise RuntimeError(f"CSV yükleme hatası: {e}")
    
    def _fix_channel_names(self):
        """Kanal adlarını düzelt"""
        if not self.channel_mapping:
            return
            
        # Mevcut kanal adlarını al
        current_names = self.raw.ch_names.copy()
        
        # Eşleme uygula
        new_names = []
        for ch_name in current_names:
            if ch_name in self.channel_mapping:
                new_names.append(self.channel_mapping[ch_name])
            else:
                new_names.append(ch_name)
                
        # Kanal adlarını güncelle
        self.raw.rename_channels(dict(zip(current_names, new_names)))
        
        if self.verbose:
            console.print(f"[blue]Kanal adları düzeltildi: {len(self.channel_mapping)} eşleme[/blue]")
    
    def _apply_montage(self, montage: Union[str, Path]):
        """Montaj uygula"""
        try:
            if isinstance(montage, str):
                if montage == "standard_1020":
                    self.montage = mne.channels.make_standard_montage('standard_1020')
                elif montage == "standard_1005":
                    self.montage = mne.channels.make_standard_montage('standard_1005')
                else:
                    # Özel montaj dosyası
                    montage_path = Path(montage)
                    if montage_path.exists():
                        if montage_path.suffix == '.loc':
                            self.montage = mne.channels.read_custom_montage(montage_path)
                        elif montage_path.suffix == '.sfp':
                            self.montage = mne.channels.read_custom_montage(montage_path)
                        else:
                            raise ValueError(f"Desteklenmeyen montaj formatı: {montage_path.suffix}")
                    else:
                        raise FileNotFoundError(f"Montaj dosyası bulunamadı: {montage_path}")
            else:
                # Path objesi
                montage_path = Path(montage)
                if montage_path.suffix == '.loc':
                    self.montage = mne.channels.read_custom_montage(montage_path)
                elif montage_path.suffix == '.sfp':
                    self.montage = mne.channels.read_custom_montage(montage_path)
                else:
                    raise ValueError(f"Desteklenmeyen montaj formatı: {montage_path.suffix}")
                    
            # Montajı uygula
            self.raw.set_montage(self.montage, on_missing='warn')
            
            if self.verbose:
                console.print(f"[green]Montaj uygulandı: {self.montage.get_positions()['ch_pos'].keys()}[/green]")
                
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]Montaj uygulanamadı: {e}[/yellow]")
    
    def get_channel_info(self) -> Dict[str, Dict]:
        """Kanal bilgilerini döndür"""
        if self.raw is None:
            return {}
            
        info = {}
        for i, ch_name in enumerate(self.raw.ch_names):
            info[ch_name] = {
                'index': i,
                'type': self.raw.get_channel_types()[i],
                'position': self.raw.get_montage().get_positions()['ch_pos'].get(ch_name, None) if self.raw.get_montage() else None
            }
        return info
    
    def detect_bipolar_channels(self) -> List[Tuple[str, str]]:
        """Bipolar kanalları otomatik tespit et"""
        bipolar_pairs = []
        ch_names = self.raw.ch_names
        
        # Yaygın bipolar kanal isimlendirme desenleri
        patterns = [
            r'^([A-Z]+)(\d+)-([A-Z]+)(\d+)$',  # F3-F4, C3-C4
            r'^([A-Z]+)(\d+)_([A-Z]+)(\d+)$',  # F3_F4, C3_C4
            r'^([A-Z]+)(\d+)-([A-Z]+)(\d+)$',  # F3-F4, C3-C4
        ]
        
        for ch_name in ch_names:
            for pattern in patterns:
                match = re.match(pattern, ch_name)
                if match:
                    ch1 = f"{match.group(1)}{match.group(2)}"
                    ch2 = f"{match.group(3)}{match.group(4)}"
                    bipolar_pairs.append((ch1, ch2))
                    break
                    
        return bipolar_pairs


def create_channel_mapping(
    current_names: List[str],
    target_names: List[str],
    fuzzy_match: bool = True
) -> Dict[str, str]:
    """
    Kanal adı eşleme sözlüğü oluştur
    
    Parameters
    ----------
    current_names : list
        Mevcut kanal adları
    target_names : list
        Hedef kanal adları
    fuzzy_match : bool
        Yakın eşleşme kullan
        
    Returns
    -------
    dict
        Kanal eşleme sözlüğü
    """
    mapping = {}
    
    if fuzzy_match:
        # Yakın eşleşme için basit algoritma
        for current in current_names:
            best_match = None
            best_score = 0
            
            for target in target_names:
                # Basit benzerlik skoru
                score = _calculate_similarity(current.upper(), target.upper())
                if score > best_score and score > 0.7:  # %70 eşik
                    best_score = score
                    best_match = target
                    
            if best_match:
                mapping[current] = best_match
    else:
        # Tam eşleşme
        for current in current_names:
            if current in target_names:
                mapping[current] = current
                
    return mapping


def _calculate_similarity(s1: str, s2: str) -> float:
    """İki string arasındaki benzerlik skorunu hesapla"""
    if s1 == s2:
        return 1.0
        
    # Levenshtein distance tabanlı benzerlik
    m, n = len(s1), len(s2)
    if m == 0:
        return 0.0
    if n == 0:
        return 0.0
        
    # Dinamik programlama ile Levenshtein distance
    d = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
                
    max_len = max(m, n)
    return 1.0 - (d[m][n] / max_len)


def load_channel_mapping(file_path: Union[str, Path]) -> Dict[str, str]:
    """JSON dosyasından kanal eşleme yükle"""
    import json
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_channel_mapping(mapping: Dict[str, str], file_path: Union[str, Path]):
    """Kanal eşleme sözlüğünü JSON dosyasına kaydet"""
    import json
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
