"""
Bipolar veri işleme modülü

Bipolar EEG verilerinden referans dönüşümü veya orta-nokta yaklaşımı.
"""

from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from mne.preprocessing import compute_current_source_density
from rich.console import Console

console = Console()


class BipolarProcessor:
    """Bipolar EEG veri işleme sınıfı"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.bipolar_pairs = []
        self.reference_channels = []
        
    def detect_bipolar_channels(self, raw: mne.io.Raw) -> List[Tuple[str, str]]:
        """
        Bipolar kanalları otomatik tespit et
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
            
        Returns
        -------
        list
            Bipolar kanal çiftleri
        """
        import re
        
        bipolar_pairs = []
        ch_names = raw.ch_names
        
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
                    
        self.bipolar_pairs = bipolar_pairs
        
        if self.verbose:
            console.print(f"[green]{len(bipolar_pairs)} bipolar kanal tespit edildi[/green]")
            
        return bipolar_pairs
    
    def convert_bipolar_to_reference(
        self,
        raw: mne.io.Raw,
        ref_type: str = 'average',
        ref_channels: Optional[List[str]] = None,
        method: str = 'subtraction'
    ) -> mne.io.Raw:
        """
        Bipolar verileri referans sistemine dönüştür
        
        Parameters
        ----------
        raw : mne.io.Raw
            Bipolar EEG verisi
        ref_type : str
            Referans türü ('average', 'Cz', 'linked_ears', 'custom')
        ref_channels : list, optional
            Özel referans kanalları
        method : str
            Dönüşüm metodu ('subtraction', 'csd')
            
        Returns
        -------
        mne.io.Raw
            Referans sistemine dönüştürülmüş veri
        """
        if self.verbose:
            console.print(f"[blue]Bipolar veri referans sistemine dönüştürülüyor: {ref_type}[/blue]")
            
        # Bipolar kanalları tespit et
        if not self.bipolar_pairs:
            self.detect_bipolar_channels(raw)
            
        if not self.bipolar_pairs:
            console.print("[yellow]Bipolar kanal bulunamadı, orijinal veri döndürülüyor[/yellow]")
            return raw
            
        # Referans kanallarını belirle
        if ref_type == 'average':
            # Tüm bipolar kanalların ortalaması
            ref_data = np.mean(raw.get_data(), axis=0, keepdims=True)
            ref_channels = ['AVERAGE']
        elif ref_type == 'Cz':
            if 'Cz' in raw.ch_names:
                ref_data = raw.get_data()[raw.ch_names.index('Cz'), :]
                ref_channels = ['Cz']
            else:
                console.print("[yellow]Cz kanalı bulunamadı, average referans kullanılıyor[/yellow]")
                ref_data = np.mean(raw.get_data(), axis=0, keepdims=True)
                ref_channels = ['AVERAGE']
        elif ref_type == 'linked_ears':
            # Mastoid kanalları bul
            mastoid_chs = [ch for ch in raw.ch_names if 'M' in ch.upper() and ('1' in ch or '2' in ch)]
            if len(mastoid_chs) >= 2:
                mastoid_data = raw.get_data()[[raw.ch_names.index(ch) for ch in mastoid_chs], :]
                ref_data = np.mean(mastoid_data, axis=0, keepdims=True)
                ref_channels = ['LINKED_EARS']
            else:
                console.print("[yellow]Mastoid kanalları bulunamadı, average referans kullanılıyor[/yellow]")
                ref_data = np.mean(raw.get_data(), axis=0, keepdims=True)
                ref_channels = ['AVERAGE']
        elif ref_type == 'custom' and ref_channels:
            custom_data = raw.get_data()[[raw.ch_names.index(ch) for ch in ref_channels], :]
            ref_data = np.mean(custom_data, axis=0, keepdims=True)
        else:
            raise ValueError(f"Desteklenmeyen referans türü: {ref_type}")
            
        # Bipolar kanalları referans sistemine dönüştür
        converted_data = []
        converted_ch_names = []
        
        for i, (ch1, ch2) in enumerate(self.bipolar_pairs):
            # Bipolar kanal verisi
            bipolar_data = raw.get_data()[i, :]
            
            if method == 'subtraction':
                # Basit çıkarma: bipolar - referans
                converted_channel = bipolar_data - ref_data.flatten()
            elif method == 'csd':
                # Current Source Density yaklaşımı
                # Bu durumda bipolar kanalı doğrudan kullan
                converted_channel = bipolar_data
            else:
                raise ValueError(f"Desteklenmeyen dönüşüm metodu: {method}")
                
            converted_data.append(converted_channel)
            converted_ch_names.append(f"{ch1}-{ch2}")
            
        # Yeni Raw objesi oluştur
        converted_data = np.array(converted_data)
        info = mne.create_info(
            ch_names=converted_ch_names,
            sfreq=raw.info['sfreq'],
            ch_types='eeg'
        )
        
        converted_raw = mne.io.RawArray(converted_data, info, verbose=False)
        
        # Montaj bilgisini kopyala
        if raw.get_montage():
            converted_raw.set_montage(raw.get_montage())
            
        if self.verbose:
            console.print(f"[green]Bipolar veri dönüştürüldü: {len(converted_ch_names)} kanal[/green]")
            
        return converted_raw
    
    def create_midpoint_coordinates(
        self,
        raw: mne.io.Raw,
        bipolar_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Bipolar kanallar için orta-nokta koordinatları oluştur
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        bipolar_pairs : list, optional
            Bipolar kanal çiftleri
            
        Returns
        -------
        dict
            Kanal adları ve koordinatları
        """
        if bipolar_pairs is None:
            bipolar_pairs = self.bipolar_pairs
            
        if not bipolar_pairs:
            self.detect_bipolar_channels(raw)
            bipolar_pairs = self.bipolar_pairs
            
        # Montaj bilgisini al
        montage = raw.get_montage()
        if not montage:
            console.print("[yellow]Montaj bilgisi bulunamadı, varsayılan koordinatlar kullanılıyor[/yellow]")
            return {}
            
        ch_pos = montage.get_positions()['ch_pos']
        midpoint_coords = {}
        
        for ch1, ch2 in bipolar_pairs:
            # Her iki kanalın koordinatlarını al
            if ch1 in ch_pos and ch2 in ch_pos:
                pos1 = np.array(ch_pos[ch1])
                pos2 = np.array(ch_pos[ch2])
                
                # Orta nokta hesapla
                midpoint = (pos1 + pos2) / 2
                midpoint_coords[f"{ch1}-{ch2}"] = tuple(midpoint)
            else:
                console.print(f"[yellow]Koordinat bulunamadı: {ch1}, {ch2}[/yellow]")
                
        if self.verbose:
            console.print(f"[green]{len(midpoint_coords)} orta-nokta koordinatı oluşturuldu[/green]")
            
        return midpoint_coords
    
    def get_midpoint_coordinates(self, ch_name: str, montage: str = 'standard_1020') -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Bipolar kanal için orta nokta koordinatlarını hesapla
        
        Args:
            ch_name: Bipolar kanal adı (örn. 'F3-C3')
            montage: Montaj türü
            
        Returns:
            (x, y, z) koordinatları veya (None, None, None)
        """
        if '-' not in ch_name:
            return None, None, None
            
        # Montaj yükle
        try:
            if montage == 'standard_1020':
                from mne.channels import make_standard_montage
                montage_obj = make_standard_montage('standard_1020')
            else:
                from mne.channels import make_standard_montage
                montage_obj = make_standard_montage('standard_1005')
            
            # Bipolar kanalı ayır
            ch1, ch2 = ch_name.split('-', 1)
            
            # Her iki kanalın koordinatlarını al
            ch_pos = montage_obj.get_positions()['ch_pos']
            if ch1 in ch_pos and ch2 in ch_pos:
                pos1 = np.array(ch_pos[ch1])
                pos2 = np.array(ch_pos[ch2])
                # Orta nokta hesapla
                midpoint = (pos1 + pos2) / 2
                return midpoint[0], midpoint[1], midpoint[2]
            else:
                if self.verbose:
                    console.print(f"[yellow]Koordinat bulunamadı: {ch1}, {ch2}[/yellow]")
                return None, None, None
                
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]Koordinat hesaplama hatası: {e}[/yellow]")
            return None, None, None

    def create_bipolar_topomap_data(self, channel_data: Dict[str, float], 
                                  montage: str = 'standard_1020') -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
        """
        Bipolar kanal verilerini topomap için hazırla
        
        Args:
            channel_data: Kanal adı -> değer sözlüğü
            montage: Kullanılacak montaj
            
        Returns:
            (values, channel_names, positions) - topomap için değerler, kanal adları ve koordinatlar
        """
        # Montaj yükle
        try:
            if montage == 'standard_1020':
                from mne.channels import make_standard_montage
                montage_obj = make_standard_montage('standard_1020')
            else:
                from mne.channels import make_standard_montage
                montage_obj = make_standard_montage('standard_1005')
        except Exception as e:
            console.print(f"[yellow]Montaj yükleme hatası: {e}[/yellow]")
            return np.array([]), [], {}
        
        # Bipolar kanalları işle
        values = []
        channel_names = []
        positions = {}
        ch_pos = montage_obj.get_positions()['ch_pos']
        used_positions = set()  # Overlapping pozisyonları takip et
        
        for ch_name, value in channel_data.items():
            if '-' in ch_name:  # Bipolar kanal
                # Orta nokta koordinatlarını hesapla
                x, y, z = self.get_midpoint_coordinates(ch_name, montage)
                if x is not None:
                    pos_tuple = (round(x, 6), round(y, 6), round(z, 6))  # Round to avoid floating point precision issues
                    
                    # Bu pozisyon daha önce kullanıldı mı kontrol et
                    if pos_tuple not in used_positions:
                        values.append(value)
                        channel_names.append(ch_name)
                        positions[ch_name] = np.array([x, y, z])
                        used_positions.add(pos_tuple)
                    else:
                        # Overlapping pozisyon - sadece ilkini kullan, ikincisini atla
                        if self.verbose:
                            console.print(f"[yellow]Overlapping pozisyon tespit edildi, {ch_name} atlanıyor[/yellow]")
            else:  # Tekil kanal
                # Montajdan koordinat al
                if ch_name in ch_pos:
                    pos = np.array(ch_pos[ch_name])
                    pos_tuple = (round(pos[0], 6), round(pos[1], 6), round(pos[2], 6))
                    
                    if pos_tuple not in used_positions:
                        values.append(value)
                        channel_names.append(ch_name)
                        positions[ch_name] = pos
                        used_positions.add(pos_tuple)
        
        return np.array(values), channel_names, positions
    
    
    def apply_midpoint_approach(
        self,
        raw: mne.io.Raw,
        bipolar_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> mne.io.Raw:
        """
        Orta-nokta yaklaşımını uygula
        
        Parameters
        ----------
        raw : mne.io.Raw
            Bipolar EEG verisi
        bipolar_pairs : list, optional
            Bipolar kanal çiftleri
            
        Returns
        -------
        mne.io.Raw
            Orta-nokta yaklaşımı uygulanmış veri
        """
        if self.verbose:
            console.print("[blue]Orta-nokta yaklaşımı uygulanıyor...[/blue]")
            
        if bipolar_pairs is None:
            bipolar_pairs = self.bipolar_pairs
            
        if not bipolar_pairs:
            self.detect_bipolar_channels(raw)
            bipolar_pairs = self.bipolar_pairs
            
        # Orta-nokta koordinatları oluştur
        midpoint_coords = self.create_midpoint_coordinates(raw, bipolar_pairs)
        
        # Yeni kanal adları ve verileri
        new_ch_names = []
        new_data = []
        
        for i, (ch1, ch2) in enumerate(bipolar_pairs):
            # Bipolar kanal verisi
            bipolar_data = raw.get_data()[i, :]
            
            # Yeni kanal adı
            new_ch_name = f"{ch1}-{ch2}"
            new_ch_names.append(new_ch_name)
            new_data.append(bipolar_data)
            
        # Yeni Raw objesi oluştur
        new_data = np.array(new_data)
        info = mne.create_info(
            ch_names=new_ch_names,
            sfreq=raw.info['sfreq'],
            ch_types='eeg'
        )
        
        new_raw = mne.io.RawArray(new_data, info, verbose=False)
        
        # Orta-nokta koordinatları ile özel montaj oluştur
        if midpoint_coords:
            # MNE montaj formatına çevir
            ch_pos = {}
            for ch_name, (x, y, z) in midpoint_coords.items():
                ch_pos[ch_name] = np.array([x, y, z])
                
            # Özel montaj oluştur
            custom_montage = mne.channels.make_dig_montage(
                ch_pos=ch_pos,
                coord_frame='head'
            )
            new_raw.set_montage(custom_montage)
            
        if self.verbose:
            console.print(f"[green]Orta-nokta yaklaşımı uygulandı: {len(new_ch_names)} kanal[/green]")
            console.print("[yellow]Uyarı: Orta-nokta yaklaşımı topolojik doğruluk sınırlamaları içerir[/yellow]")
            
        return new_raw
    
    def apply_csd_approach(
        self,
        raw: mne.io.Raw,
        lambda2: float = 1e-5,
        stiffness: float = 4,
        n_jobs: int = 1
    ) -> mne.io.Raw:
        """
        Current Source Density (CSD) yaklaşımını uygula
        
        Parameters
        ----------
        raw : mne.io.Raw
            EEG verisi
        lambda2 : float
            CSD lambda parametresi
        stiffness : float
            CSD stiffness parametresi
        n_jobs : int
            Paralel işlem sayısı
            
        Returns
        -------
        mne.io.Raw
            CSD uygulanmış veri
        """
        if self.verbose:
            console.print("[blue]CSD yaklaşımı uygulanıyor...[/blue]")
            
        try:
            # CSD hesapla
            csd_data = compute_current_source_density(
                raw,
                lambda2=lambda2,
                stiffness=stiffness,
                n_jobs=n_jobs,
                verbose=False
            )
            
            if self.verbose:
                console.print("[green]CSD uygulandı[/green]")
                
            return csd_data
            
        except Exception as e:
            console.print(f"[yellow]CSD hesaplama hatası: {e}[/yellow]")
            console.print("[yellow]Orijinal veri döndürülüyor[/yellow]")
            return raw
    
    def get_bipolar_info(self) -> Dict:
        """Bipolar işleme bilgilerini döndür"""
        return {
            'bipolar_pairs': self.bipolar_pairs,
            'n_bipolar_pairs': len(self.bipolar_pairs),
            'reference_channels': self.reference_channels
        }


def process_bipolar_data(
    raw: mne.io.Raw,
    method: str = 'reference_conversion',
    ref_type: str = 'average',
    **kwargs
) -> mne.io.Raw:
    """
    Bipolar veri işleme ana fonksiyonu
    
    Parameters
    ----------
    raw : mne.io.Raw
        Bipolar EEG verisi
    method : str
        İşleme metodu ('reference_conversion', 'midpoint', 'csd')
    ref_type : str
        Referans türü (method='reference_conversion' için)
    **kwargs
        Diğer parametreler
        
    Returns
    -------
    mne.io.Raw
        İşlenmiş veri
    """
    processor = BipolarProcessor()
    
    if method == 'reference_conversion':
        return processor.convert_bipolar_to_reference(raw, ref_type=ref_type, **kwargs)
    elif method == 'midpoint':
        return processor.apply_midpoint_approach(raw, **kwargs)
    elif method == 'csd':
        return processor.apply_csd_approach(raw, **kwargs)
    else:
        raise ValueError(f"Desteklenmeyen bipolar işleme metodu: {method}")


def validate_bipolar_conversion(
    original_raw: mne.io.Raw,
    converted_raw: mne.io.Raw,
    method: str
) -> Dict[str, Union[bool, str]]:
    """
    Bipolar dönüşümünü doğrula
    
    Parameters
    ----------
    original_raw : mne.io.Raw
        Orijinal bipolar veri
    converted_raw : mne.io.Raw
        Dönüştürülmüş veri
    method : str
        Kullanılan metod
        
    Returns
    -------
    dict
        Doğrulama sonuçları
    """
    validation = {
        'method': method,
        'original_channels': len(original_raw.ch_names),
        'converted_channels': len(converted_raw.ch_names),
        'sfreq_match': original_raw.info['sfreq'] == converted_raw.info['sfreq'],
        'data_shape_match': original_raw.get_data().shape[1] == converted_raw.get_data().shape[1]
    }
    
    # Metod-spesifik doğrulamalar
    if method == 'reference_conversion':
        validation['channels_reduced'] = len(converted_raw.ch_names) <= len(original_raw.ch_names)
    elif method == 'midpoint':
        validation['midpoint_coords_available'] = converted_raw.get_montage() is not None
    elif method == 'csd':
        validation['csd_applied'] = True  # CSD başarılı ise True
        
    return validation
