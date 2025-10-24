"""
Dışa aktarma modülü

SVG/PNG/PDF çıktıları ve JSON/YAML metadata.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rich.console import Console

console = Console()


class EEGExporter:
    """EEG veri dışa aktarma sınıfı"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.export_metadata = {}
        
    def export_figure(
        self,
        fig: plt.Figure,
        file_path: Union[str, Path],
        format: str = 'svg',
        dpi: int = 300,
        bbox_inches: str = 'tight',
        pad_inches: float = 0.1,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Figürü dışa aktar
        
        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figürü
        file_path : str or Path
            Dosya yolu
        format : str
            Dosya formatı ('svg', 'png', 'pdf', 'eps')
        dpi : int
            Çözünürlük (raster formatlar için)
        bbox_inches : str
            Bounding box ayarı
        pad_inches : float
            Padding
        metadata : dict, optional
            Metadata bilgileri
            
        Returns
        -------
        bool
            Başarı durumu
        """
        file_path = Path(file_path)
        
        # Dosya uzantısını kontrol et
        if not file_path.suffix:
            file_path = file_path.with_suffix(f'.{format}')
            
        # Dizin oluştur
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Figürü kaydet
            fig.savefig(
                file_path,
                format=format,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor='white',
                edgecolor='none'
            )
            
            if self.verbose:
                console.print(f"[green]Figür kaydedildi: {file_path}[/green]")
                
            # Metadata kaydet
            if metadata:
                metadata_path = file_path.with_suffix('.json')
                self.export_metadata_file(metadata, metadata_path)
                
            return True
            
        except Exception as e:
            console.print(f"[red]Figür kaydetme hatası: {e}[/red]")
            return False
    
    def export_metadata_file(
        self,
        metadata: Dict[str, Any],
        file_path: Union[str, Path],
        format: str = 'json'
    ) -> bool:
        """
        Metadata dosyası oluştur
        
        Parameters
        ----------
        metadata : dict
            Metadata bilgileri
        file_path : str or Path
            Dosya yolu
        format : str
            Dosya formatı ('json', 'yaml')
            
        Returns
        -------
        bool
            Başarı durumu
        """
        file_path = Path(file_path)
        
        # Dosya uzantısını kontrol et
        if not file_path.suffix:
            file_path = file_path.with_suffix(f'.{format}')
            
        # Dizin oluştur
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            elif format.lower() in ['yaml', 'yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Desteklenmeyen format: {format}")
                
            if self.verbose:
                console.print(f"[green]Metadata kaydedildi: {file_path}[/green]")
                
            return True
            
        except Exception as e:
            console.print(f"[red]Metadata kaydetme hatası: {e}[/red]")
            return False
    
    def create_analysis_metadata(
        self,
        input_file: str,
        montage: str,
        segments: Dict[str, List[float]],
        metrics: List[str],
        statistics: Optional[Dict] = None,
        visualization: Optional[Dict] = None,
        preprocessing: Optional[Dict] = None,
        bipolar_processing: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analiz metadata'sı oluştur
        
        Parameters
        ----------
        input_file : str
            Girdi dosyası
        montage : str
            Montaj bilgisi
        segments : dict
            Segment bilgileri
        metrics : list
            Metrik listesi
        statistics : dict, optional
            İstatistik bilgileri
        visualization : dict, optional
            Görselleştirme bilgileri
        preprocessing : dict, optional
            Ön işleme bilgileri
        bipolar_processing : dict, optional
            Bipolar işleme bilgileri
            
        Returns
        -------
        dict
            Metadata sözlüğü
        """
        metadata = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'software_version': 'eeg-topomap-lab-0.1.0',
                'input_file': str(input_file),
                'montage': montage
            },
            'segments': segments,
            'metrics': metrics,
            'statistics': statistics or {},
            'visualization': visualization or {},
            'preprocessing': preprocessing or {},
            'bipolar_processing': bipolar_processing or {}
        }
        
        return metadata
    
    def export_metrics_table(
        self,
        metrics_data: Dict[str, np.ndarray],
        channel_names: List[str],
        file_path: Union[str, Path],
        format: str = 'csv'
    ) -> bool:
        """
        Metrik tablosunu dışa aktar
        
        Parameters
        ----------
        metrics_data : dict
            Metrik verileri
        channel_names : list
            Kanal adları
        file_path : str or Path
            Dosya yolu
        format : str
            Dosya formatı ('csv', 'excel', 'json')
            
        Returns
        -------
        bool
            Başarı durumu
        """
        file_path = Path(file_path)
        
        # DataFrame oluştur
        df_data = {'channel': channel_names}
        
        for metric_name, values in metrics_data.items():
            if isinstance(values, np.ndarray) and values.ndim == 1:
                df_data[metric_name] = values
            elif isinstance(values, np.ndarray) and values.ndim == 2:
                # 2D array ise ortalama al
                df_data[metric_name] = np.mean(values, axis=1)
                
        df = pd.DataFrame(df_data)
        
        # Dizin oluştur
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == 'csv':
                df.to_csv(file_path, index=False)
            elif format.lower() == 'excel':
                df.to_excel(file_path, index=False)
            elif format.lower() == 'json':
                df.to_json(file_path, orient='records', indent=2)
            else:
                raise ValueError(f"Desteklenmeyen format: {format}")
                
            if self.verbose:
                console.print(f"[green]Metrik tablosu kaydedildi: {file_path}[/green]")
                
            return True
            
        except Exception as e:
            console.print(f"[red]Metrik tablosu kaydetme hatası: {e}[/red]")
            return False
    
    def export_statistical_results(
        self,
        test_results: Dict[str, Any],
        channel_names: List[str],
        file_path: Union[str, Path],
        format: str = 'csv'
    ) -> bool:
        """
        İstatistiksel sonuçları dışa aktar
        
        Parameters
        ----------
        test_results : dict
            Test sonuçları
        channel_names : list
            Kanal adları
        file_path : str or Path
            Dosya yolu
        format : str
            Dosya formatı
            
        Returns
        -------
        bool
            Başarı durumu
        """
        file_path = Path(file_path)
        
        # DataFrame oluştur
        df_data = {'channel': channel_names}
        
        # Test sonuçlarını ekle
        if 'p_values' in test_results:
            df_data['p_value'] = test_results['p_values']
            
        if 'statistics' in test_results:
            df_data['statistic'] = test_results['statistics']
            
        if 'effect_sizes' in test_results:
            df_data['effect_size'] = test_results['effect_sizes']
            
        if 'corrected_p_values' in test_results:
            df_data['corrected_p_value'] = test_results['corrected_p_values']
            df_data['significant'] = test_results['significant']
            
        df = pd.DataFrame(df_data)
        
        # Dizin oluştur
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == 'csv':
                df.to_csv(file_path, index=False)
            elif format.lower() == 'excel':
                df.to_excel(file_path, index=False)
            elif format.lower() == 'json':
                df.to_json(file_path, orient='records', indent=2)
            else:
                raise ValueError(f"Desteklenmeyen format: {format}")
                
            if self.verbose:
                console.print(f"[green]İstatistiksel sonuçlar kaydedildi: {file_path}[/green]")
                
            return True
            
        except Exception as e:
            console.print(f"[red]İstatistiksel sonuçlar kaydetme hatası: {e}[/red]")
            return False
    
    def export_config_file(
        self,
        config: Dict[str, Any],
        file_path: Union[str, Path],
        format: str = 'yaml'
    ) -> bool:
        """
        Konfigürasyon dosyası oluştur
        
        Parameters
        ----------
        config : dict
            Konfigürasyon bilgileri
        file_path : str or Path
            Dosya yolu
        format : str
            Dosya formatı ('yaml', 'json')
            
        Returns
        -------
        bool
            Başarı durumu
        """
        file_path = Path(file_path)
        
        # Dizin oluştur
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() in ['yaml', 'yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False, default=str)
            else:
                raise ValueError(f"Desteklenmeyen format: {format}")
                
            if self.verbose:
                console.print(f"[green]Konfigürasyon kaydedildi: {file_path}[/green]")
                
            return True
            
        except Exception as e:
            console.print(f"[red]Konfigürasyon kaydetme hatası: {e}[/red]")
            return False
    
    def create_publication_caption(
        self,
        analysis_type: str,
        metrics: List[str],
        segments: Dict[str, List[float]],
        statistics: Optional[Dict] = None,
        n_subjects: Optional[int] = None
    ) -> str:
        """
        Yayın caption'ı oluştur
        
        Parameters
        ----------
        analysis_type : str
            Analiz türü
        metrics : list
            Metrik listesi
        segments : dict
            Segment bilgileri
        statistics : dict, optional
            İstatistik bilgileri
        n_subjects : int, optional
            Katılımcı sayısı
            
        Returns
        -------
        str
            Caption metni
        """
        caption_parts = []
        
        # Analiz türü
        if analysis_type == 'single_group':
            caption_parts.append("Tek grup EEG analizi")
        elif analysis_type == 'two_group':
            caption_parts.append("İki grup EEG karşılaştırması")
        else:
            caption_parts.append(f"{analysis_type} EEG analizi")
            
        # Metrikler
        metric_names = {
            'rms': 'RMS',
            'peak_to_peak': 'Tepe-tepe',
            'mean': 'Ortalama',
            'psd': 'Güç spektral yoğunluğu',
            'band_power': 'Bant gücü',
            'dfa': 'DFA',
            'lempel_ziv': 'Lempel-Ziv kompleksitesi',
            'higuchi_fd': 'Higuchi fraktal boyutu',
            'permutation_entropy': 'Permutasyon entropisi'
        }
        
        metric_text = []
        for metric in metrics:
            if metric in metric_names:
                metric_text.append(metric_names[metric])
            else:
                metric_text.append(metric)
                
        caption_parts.append(f"Metrikler: {', '.join(metric_text)}")
        
        # Segmentler
        segment_text = []
        for seg_name, (start, end) in segments.items():
            segment_text.append(f"{seg_name} ({start}-{end}s)")
        caption_parts.append(f"Segmentler: {', '.join(segment_text)}")
        
        # İstatistikler
        if statistics:
            if 'test_type' in statistics:
                test_names = {
                    'ttest': 't-testi',
                    'mannwhitney': 'Mann-Whitney U testi',
                    'wilcoxon': 'Wilcoxon signed-rank testi'
                }
                test_name = test_names.get(statistics['test_type'], statistics['test_type'])
                caption_parts.append(f"İstatistik: {test_name}")
                
            if 'correction_method' in statistics:
                correction_names = {
                    'fdr': 'FDR düzeltmesi',
                    'bonferroni': 'Bonferroni düzeltmesi',
                    'holm': 'Holm düzeltmesi'
                }
                correction_name = correction_names.get(statistics['correction_method'], statistics['correction_method'])
                caption_parts.append(f"Düzeltme: {correction_name}")
                
        # Katılımcı sayısı
        if n_subjects:
            caption_parts.append(f"Katılımcı sayısı: {n_subjects}")
            
        return ". ".join(caption_parts) + "."
    
    def export_batch(
        self,
        figures: List[plt.Figure],
        base_path: Union[str, Path],
        formats: List[str] = ['svg', 'png'],
        metadata: Optional[Dict] = None
    ) -> Dict[str, bool]:
        """
        Toplu dışa aktarma
        
        Parameters
        ----------
        figures : list
            Figür listesi
        base_path : str or Path
            Temel dosya yolu
        formats : list
            Dosya formatları
        metadata : dict, optional
            Metadata bilgileri
            
        Returns
        -------
        dict
            Başarı durumları
        """
        base_path = Path(base_path)
        results = {}
        
        for i, fig in enumerate(figures):
            for format in formats:
                file_path = base_path / f"figure_{i+1:02d}.{format}"
                success = self.export_figure(fig, file_path, format=format, metadata=metadata)
                results[str(file_path)] = success
                
        return results


def export_analysis_results(
    figures: List[plt.Figure],
    metrics_data: Dict[str, np.ndarray],
    test_results: Optional[Dict] = None,
    output_dir: Union[str, Path] = "output",
    formats: List[str] = ['svg', 'png'],
    metadata: Optional[Dict] = None
) -> bool:
    """
    Analiz sonuçlarını toplu dışa aktar
    
    Parameters
    ----------
    figures : list
        Figür listesi
    metrics_data : dict
        Metrik verileri
    test_results : dict, optional
        Test sonuçları
    output_dir : str or Path
        Çıktı dizini
    formats : list
        Dosya formatları
    metadata : dict, optional
        Metadata bilgileri
        
    Returns
    -------
    bool
        Başarı durumu
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exporter = EEGExporter()
    
    # Figürleri kaydet
    for i, fig in enumerate(figures):
        for format in formats:
            file_path = output_dir / f"topomap_{i+1:02d}.{format}"
            exporter.export_figure(fig, file_path, format=format)
            
    # Metadata kaydet
    if metadata:
        metadata_path = output_dir / "analysis_metadata.json"
        exporter.export_metadata_file(metadata, metadata_path)
        
    return True
