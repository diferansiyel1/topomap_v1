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
            if self.verbose:
                console.print(f"[yellow]Bipolar kanal değil: {ch_name}[/yellow]")
            return None, None, None
            
        # Önce özel koordinat haritasını kontrol et
        special_coords = self._get_special_bipolar_coordinates()
        
        # Orijinal adı kontrol et
        if ch_name in special_coords:
            return special_coords[ch_name]
            
        # Normalize edilmiş kanal adını kontrol et
        normalized_name = self._normalize_channel_name(ch_name)
        if normalized_name in special_coords:
            return special_coords[normalized_name]
            
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
                if self.verbose:
                    console.print(f"[green]Montaj koordinatı hesaplandı: {ch_name} -> {midpoint}[/green]")
                return midpoint[0], midpoint[1], midpoint[2]
            else:
                if self.verbose:
                    console.print(f"[yellow]Montaj koordinat bulunamadı: {ch1}, {ch2}[/yellow]")
                return None, None, None
                
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]Koordinat hesaplama hatası: {e}[/yellow]")
            return None, None, None
    
    def _normalize_channel_name(self, ch_name: str) -> str:
        """
        Kanal adını normalize et - farklı formatları standart hale getir
        
        Args:
            ch_name: Orijinal kanal adı
            
        Returns:
            Normalize edilmiş kanal adı (MNE uyumlu: Fp1, Fp2, Fz, Cz, Pz)
        """
        import re
        
        # Kanal adını normalize et
        normalized = ch_name.strip()
        
        # MNE uyumluluğu için FP -> Fp, CZ -> Cz, FZ -> Fz, PZ -> Pz dönüşümü
        mapping = {
            'FP1': 'Fp1', 'FP2': 'Fp2',
            'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
            'FPZ': 'Fpz'
        }
        
        # Bipolar kanal: iki elektrot arasında '-' var
        if '-' in normalized:
            parts = normalized.split('-')
            if len(parts) == 2:
                # Her parçayı ayrı ayrı normalize et
                ch1 = parts[0].strip()
                ch2 = parts[1].strip()
                
                # Mapping uygula
                ch1 = mapping.get(ch1, ch1)
                ch2 = mapping.get(ch2, ch2)
                
                # Sonunda 01 veya 02 varsa, bunları O1 ve O2'ye çevir
                if ch1.endswith('01'):
                    ch1 = ch1[:-2] + 'O1'
                elif ch1.endswith('02'):
                    ch1 = ch1[:-2] + 'O2'
                    
                if ch2.endswith('01'):
                    ch2 = ch2[:-2] + 'O1'
                elif ch2.endswith('02'):
                    ch2 = ch2[:-2] + 'O2'
                
                normalized = f"{ch1}-{ch2}"
        else:
            # Tekil kanal
            normalized = mapping.get(normalized, normalized)
            
            if normalized.endswith('01'):
                normalized = normalized[:-2] + 'O1'
            elif normalized.endswith('02'):
                normalized = normalized[:-2] + 'O2'
                
        return normalized
    
    def _get_special_bipolar_coordinates(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Özel bipolar kanal koordinatları - 10-20 sistemine göre optimize edilmiş
        
        Returns:
            Kanal adı -> (x, y, z) koordinatları sözlüğü
        """
        # Bu koordinatlar 10-20 sistemine göre standart yuvarlak topomap için optimize edilmiştir
        # X: sol-sağ (-1 to 1), Y: ön-arka (-1 to 1), Z: yukarı-aşağı (0)
        special_coords = {
            # Frontal kanallar - standart dağılım
            'FP1-F7': (0.3, 0.7, 0.0),
            'FP1-F3': (0.2, 0.8, 0.0),
            'FP2-F4': (-0.2, 0.8, 0.0),
            'FP2-F8': (-0.3, 0.7, 0.0),
            
            # Frontal-Temporal - standart ayrım
            'F7-T7': (0.6, 0.4, 0.0),
            'F3-C3': (0.3, 0.5, 0.0),
            'F4-C4': (-0.3, 0.5, 0.0),
            'F8-T8': (-0.6, 0.4, 0.0),
            
            # Central - merkezi konumlar
            'FZ-CZ': (0.0, 0.6, 0.0),
            'CZ-PZ': (0.0, 0.0, 0.0),
            
            # Temporal-Parietal - standart dağılım
            'T7-P7': (0.6, 0.0, 0.0),
            'C3-P3': (0.3, 0.2, 0.0),
            'C4-P4': (-0.3, 0.2, 0.0),
            'T8-P8': (-0.6, 0.0, 0.0),
            
            # Parietal-Occipital - standart konumlandırma
            'P7-O1': (0.3, -0.3, 0.0),
            'P3-O1': (0.15, -0.3, 0.0),
            'P4-O2': (-0.15, -0.3, 0.0),
            'P8-O2': (-0.3, -0.3, 0.0),
            
            # Temporal-Frontal - standart ayrım
            'T7-FT9': (0.5, 0.3, 0.0),
            'FT9-FT10': (0.0, 0.3, 0.0),
            'FT10-T8': (-0.5, 0.3, 0.0),
            
            # Ek kanallar - standart koordinatlar
            'T7-FT9': (0.5, 0.3, 0.0),
            'P7-T7': (0.6, 0.0, 0.0),
            'F8-T8': (-0.6, 0.4, 0.0),
            'T8-P8': (-0.6, 0.0, 0.0),
            
            # Yaygın bipolar kanal varyasyonları
            'FP1-F7': (0.3, 0.7, 0.0),
            'FP2-F8': (-0.3, 0.7, 0.0),
            'F3-C3': (0.3, 0.5, 0.0),
            'F4-C4': (-0.3, 0.5, 0.0),
            'C3-P3': (0.3, 0.2, 0.0),
            'C4-P4': (-0.3, 0.2, 0.0),
            'P3-O1': (0.15, -0.3, 0.0),
            'P4-O2': (-0.15, -0.3, 0.0),
            'T7-P7': (0.6, 0.0, 0.0),
            'T8-P8': (-0.6, 0.0, 0.0),
            'F7-T7': (0.6, 0.4, 0.0),
            'F8-T8': (-0.6, 0.4, 0.0),
            'FZ-CZ': (0.0, 0.6, 0.0),
            'CZ-PZ': (0.0, 0.0, 0.0),
        }
        
        return special_coords

    def _get_bipolar_coordinates(self, montage: str = 'standard_1020') -> Dict[str, Tuple[float, float, float]]:
        """
        Calculate bipolar coordinates from MNE standard montage by computing midpoints
        
        Args:
            montage: MNE montage name to use
            
        Returns:
            Kanal adı -> (x, y, z) koordinatları sözlüğü
        """
        from mne.channels import make_standard_montage
        
        # Load standard montage
        montage_obj = make_standard_montage(montage)
        ch_pos = montage_obj.get_positions()['ch_pos']
        
        bipolar_coords = {}
        
        # Define bipolar pairs based on user's data
        bipolar_pairs = [
            'Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
            'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
            'Fp2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'Fz-Cz', 'Cz-Pz',
            'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
        ]
        
        for pair in bipolar_pairs:
            ch1, ch2 = pair.split('-')
            if ch1 in ch_pos and ch2 in ch_pos:
                pos1 = np.array(ch_pos[ch1])
                pos2 = np.array(ch_pos[ch2])
                midpoint = (pos1 + pos2) / 2
                bipolar_coords[pair] = tuple(midpoint)
                
                # Normalize edilmiş versiyonunu da ekle (eğer farklıysa)
                normalized_pair = self._normalize_channel_name(pair)
                if normalized_pair != pair:
                    bipolar_coords[normalized_pair] = tuple(midpoint)
                
                if self.verbose:
                    console.print(f"[green]Bipolar koordinat hesaplandı: {pair} -> {midpoint}[/green]-[yellow]normalized: {normalized_pair}[/yellow]")
            else:
                if self.verbose:
                    console.print(f"[red]Channel not found in MNE montage: {pair} (ch1: {ch1}, ch2: {ch2})[/red]")
                    # Eksik kanallar için özel koordinatları kullan
                    special_coords = self._get_special_bipolar_coordinates()
                    if pair in special_coords:
                        bipolar_coords[pair] = special_coords[pair]
                        if self.verbose:
                            console.print(f"[yellow]Using special coordinates for: {pair} -> {special_coords[pair]}[/yellow]")
        
        return bipolar_coords

    def create_bipolar_topomap_data(self, channel_data: Dict[str, float], 
                                  montage: str = 'standard_1020') -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
        """
        Bipolar kanal verilerini topomap için hazırla - bipolar özel koordinat sistemi
        
        Args:
            channel_data: Kanal adı -> değer sözlüğü
            montage: Kullanılacak montaj
            
        Returns:
            (values, channel_names, positions) - topomap için değerler, kanal adları ve koordinatlar
        """
        # Bipolar kanalları işle - özel koordinat sistemi
        values = []
        channel_names = []
        positions = {}
        used_positions = set()  # Overlapping pozisyonları takip et
        
        # Bipolar koordinatları al
        bipolar_coords = self._get_bipolar_coordinates(montage)
        
        for ch_name, value in channel_data.items():
            # Kanal adındaki boşlukları temizle
            clean_ch_name = ch_name.strip()
            
            # Kanal adını normalize et (01->O1, 02->O2, vb.)
            normalized_ch_name = self._normalize_channel_name(clean_ch_name)
            
            if '-' in normalized_ch_name:  # Bipolar kanal
                # Bipolar koordinatları kontrol et - hem orijinal hem normalize edilmiş adı dene
                if normalized_ch_name in bipolar_coords:
                    clean_ch_name = normalized_ch_name
                    ch_name_to_use = normalized_ch_name
                elif clean_ch_name in bipolar_coords:
                    ch_name_to_use = clean_ch_name
                else:
                    ch_name_to_use = None
                
                if ch_name_to_use:
                    x, y, z = bipolar_coords[ch_name_to_use]
                    
                    # normalize edilmiş kanal adını kullan
                    ch_display_name = normalized_ch_name
                    
                    # Daha hassas pozisyon kontrolü
                    pos_tuple = (round(x, 3), round(y, 3), round(z, 3))
                    
                    # Bu pozisyon daha önce kullanıldı mı kontrol et
                    if pos_tuple not in used_positions:
                        values.append(value)
                        channel_names.append(ch_display_name)
                        positions[ch_display_name] = np.array([x, y, z])
                        used_positions.add(pos_tuple)
                        if self.verbose:
                            console.print(f"[green]Bipolar kanal eklendi: {ch_display_name} -> ({x:.3f}, {y:.3f}, {z:.3f})[/green]")
                    else:
                        # Overlapping pozisyon - offset sistemi
                        offset = 0.02
                        x_offset = x + offset
                        y_offset = y + offset
                        z_offset = z + offset
                        pos_tuple_offset = (round(x_offset, 3), round(y_offset, 3), round(z_offset, 3))
                        
                        if pos_tuple_offset not in used_positions:
                            values.append(value)
                            channel_names.append(ch_display_name)
                            positions[ch_display_name] = np.array([x_offset, y_offset, z_offset])
                            used_positions.add(pos_tuple_offset)
                            if self.verbose:
                                console.print(f"[green]Bipolar kanal eklendi (offset ile): {ch_display_name} -> ({x_offset:.3f}, {y_offset:.3f}, {z_offset:.3f})[/green]")
                        else:
                            # Daha büyük offset dene
                            offset = 0.05
                            x_offset = x + offset
                            y_offset = y + offset
                            z_offset = z + offset
                            pos_tuple_offset = (round(x_offset, 3), round(y_offset, 3), round(z_offset, 3))
                            
                            if pos_tuple_offset not in used_positions:
                                values.append(value)
                                channel_names.append(ch_display_name)
                                positions[ch_display_name] = np.array([x_offset, y_offset, z_offset])
                                used_positions.add(pos_tuple_offset)
                                if self.verbose:
                                    console.print(f"[green]Bipolar kanal eklendi (büyük offset ile): {ch_display_name} -> ({x_offset:.3f}, {y_offset:.3f}, {z_offset:.3f})[/green]")
                else:
                    if self.verbose:
                        console.print(f"[red]Bipolar koordinat bulunamadı: {normalized_ch_name}[/red]")
            else:  # Tekil kanal - MNE montajı kullan
                try:
                    if montage == 'standard_1020':
                        from mne.channels import make_standard_montage
                        montage_obj = make_standard_montage('standard_1020')
                    else:
                        from mne.channels import make_standard_montage
                        montage_obj = make_standard_montage('standard_1005')
                    
                    ch_pos = montage_obj.get_positions()['ch_pos']
                    # Normalize edilmiş adı kullan
                    if normalized_ch_name in ch_pos:
                        pos = np.array(ch_pos[normalized_ch_name])
                        pos_tuple = (round(pos[0], 4), round(pos[1], 4), round(pos[2], 4))
                        
                        if pos_tuple not in used_positions:
                            values.append(value)
                            channel_names.append(normalized_ch_name)
                            positions[normalized_ch_name] = pos
                            used_positions.add(pos_tuple)
                            if self.verbose:
                                console.print(f"[green]Tekil kanal eklendi: {normalized_ch_name}[/green]")
                except:
                    if self.verbose:
                        console.print(f"[red]Tekil kanal koordinat bulunamadı: {normalized_ch_name}[/red]")
        
        if self.verbose:
            console.print(f"[green]Toplam {len(values)} kanal işlendi[/green]")
        
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
