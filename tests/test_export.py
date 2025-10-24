"""
Dışa aktarma modülü testleri
"""

import json
import tempfile
from pathlib import Path
import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_topomap_lab import export


class TestEEGExporter:
    """EEGExporter test sınıfı"""
    
    def setup_method(self):
        """Test kurulumu"""
        self.exporter = export.EEGExporter(verbose=False)
        
        # Test figürü oluştur
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.ax.plot([1, 2, 3], [1, 4, 2])
        self.ax.set_title("Test Figure")
    
    def teardown_method(self):
        """Test temizliği"""
        plt.close(self.fig)
    
    def test_export_figure_svg(self):
        """SVG formatında figür dışa aktarma testi"""
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.exporter.export_figure(self.fig, tmp_path, format='svg')
            
            assert success == True
            assert Path(tmp_path).exists()
            assert Path(tmp_path).stat().st_size > 0
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_figure_png(self):
        """PNG formatında figür dışa aktarma testi"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.exporter.export_figure(self.fig, tmp_path, format='png', dpi=150)
            
            assert success == True
            assert Path(tmp_path).exists()
            assert Path(tmp_path).stat().st_size > 0
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_figure_pdf(self):
        """PDF formatında figür dışa aktarma testi"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.exporter.export_figure(self.fig, tmp_path, format='pdf')
            
            assert success == True
            assert Path(tmp_path).exists()
            assert Path(tmp_path).stat().st_size > 0
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_metadata_json(self):
        """JSON formatında metadata dışa aktarma testi"""
        metadata = {
            'analysis_info': {
                'timestamp': '2024-01-01T00:00:00',
                'software_version': 'eeg-topomap-lab-0.1.0'
            },
            'metrics': ['dfa', 'alpha_power'],
            'segments': {'preiktal': [300, 600], 'interiktal': [1200, 1500]}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.exporter.export_metadata_file(metadata, tmp_path, format='json')
            
            assert success == True
            assert Path(tmp_path).exists()
            
            # JSON dosyasını oku ve doğrula
            with open(tmp_path, 'r', encoding='utf-8') as f:
                loaded_metadata = json.load(f)
            
            assert loaded_metadata == metadata
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_metadata_yaml(self):
        """YAML formatında metadata dışa aktarma testi"""
        metadata = {
            'analysis_info': {
                'timestamp': '2024-01-01T00:00:00',
                'software_version': 'eeg-topomap-lab-0.1.0'
            },
            'metrics': ['dfa', 'alpha_power'],
            'segments': {'preiktal': [300, 600], 'interiktal': [1200, 1500]}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.exporter.export_metadata_file(metadata, tmp_path, format='yaml')
            
            assert success == True
            assert Path(tmp_path).exists()
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_create_analysis_metadata(self):
        """Analiz metadata oluşturma testi"""
        input_file = "test.edf"
        montage = "standard_1020"
        segments = {"preiktal": [300, 600], "interiktal": [1200, 1500]}
        metrics = ["dfa", "alpha_power"]
        statistics = {"test_type": "ttest", "fdr": 0.05}
        visualization = {"cmap": "viridis", "vmin": 0.5, "vmax": 1.2}
        preprocessing = {"l_freq": 1.0, "h_freq": 40.0}
        bipolar_processing = {"method": "reference_conversion"}
        
        metadata = self.exporter.create_analysis_metadata(
            input_file, montage, segments, metrics,
            statistics, visualization, preprocessing, bipolar_processing
        )
        
        assert isinstance(metadata, dict)
        assert 'analysis_info' in metadata
        assert 'segments' in metadata
        assert 'metrics' in metadata
        assert 'statistics' in metadata
        assert 'visualization' in metadata
        assert 'preprocessing' in metadata
        assert 'bipolar_processing' in metadata
        
        assert metadata['analysis_info']['input_file'] == input_file
        assert metadata['segments'] == segments
        assert metadata['metrics'] == metrics
    
    def test_export_metrics_table_csv(self):
        """CSV formatında metrik tablosu dışa aktarma testi"""
        metrics_data = {
            'rms': np.random.randn(19),
            'mean': np.random.randn(19),
            'dfa': np.random.randn(19)
        }
        channel_names = [f'Ch{i}' for i in range(19)]
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.exporter.export_metrics_table(
                metrics_data, channel_names, tmp_path, format='csv'
            )
            
            assert success == True
            assert Path(tmp_path).exists()
            
            # CSV dosyasını oku ve doğrula
            df = pd.read_csv(tmp_path)
            assert len(df) == 19
            assert 'channel' in df.columns
            assert 'rms' in df.columns
            assert 'mean' in df.columns
            assert 'dfa' in df.columns
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_metrics_table_excel(self):
        """Excel formatında metrik tablosu dışa aktarma testi"""
        metrics_data = {
            'rms': np.random.randn(19),
            'mean': np.random.randn(19)
        }
        channel_names = [f'Ch{i}' for i in range(19)]
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.exporter.export_metrics_table(
                metrics_data, channel_names, tmp_path, format='excel'
            )
            
            assert success == True
            assert Path(tmp_path).exists()
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_statistical_results(self):
        """İstatistiksel sonuçları dışa aktarma testi"""
        test_results = {
            'p_values': np.random.uniform(0, 1, 19),
            'statistics': np.random.randn(19),
            'effect_sizes': np.random.randn(19),
            'corrected_p_values': np.random.uniform(0, 1, 19),
            'significant': np.random.choice([True, False], 19)
        }
        channel_names = [f'Ch{i}' for i in range(19)]
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.exporter.export_statistical_results(
                test_results, channel_names, tmp_path, format='csv'
            )
            
            assert success == True
            assert Path(tmp_path).exists()
            
            # CSV dosyasını oku ve doğrula
            df = pd.read_csv(tmp_path)
            assert len(df) == 19
            assert 'channel' in df.columns
            assert 'p_value' in df.columns
            assert 'statistic' in df.columns
            assert 'effect_size' in df.columns
            assert 'corrected_p_value' in df.columns
            assert 'significant' in df.columns
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_config_file(self):
        """Konfigürasyon dosyası dışa aktarma testi"""
        config = {
            'input': {'file': 'test.edf', 'montage': 'standard_1020'},
            'segments': {'preiktal': [300, 600], 'interiktal': [1200, 1500]},
            'metrics': {'type': 'dfa', 'dfa_min': 10, 'dfa_max': 100},
            'statistics': {'compare': ['preiktal', 'interiktal'], 'fdr': 0.05}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.exporter.export_config_file(config, tmp_path, format='yaml')
            
            assert success == True
            assert Path(tmp_path).exists()
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_create_publication_caption(self):
        """Yayın caption oluşturma testi"""
        analysis_type = "two_group"
        metrics = ["dfa", "alpha_power"]
        segments = {"preiktal": [300, 600], "interiktal": [1200, 1500]}
        statistics = {"test_type": "ttest", "correction_method": "fdr"}
        n_subjects = 20
        
        caption = self.exporter.create_publication_caption(
            analysis_type, metrics, segments, statistics, n_subjects
        )
        
        assert isinstance(caption, str)
        assert len(caption) > 0
        assert "İki grup EEG karşılaştırması" in caption
        assert "DFA" in caption
        assert "preiktal" in caption
        assert "interiktal" in caption
        assert "t-testi" in caption
        assert "FDR düzeltmesi" in caption
        assert "20" in caption
    
    def test_export_batch(self):
        """Toplu dışa aktarma testi"""
        # Test figürleri oluştur
        figs = []
        for i in range(3):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title(f"Test Figure {i+1}")
            figs.append(fig)
        
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                base_path = Path(tmp_dir) / "test_output"
                formats = ['svg', 'png']
                metadata = {'test': 'metadata'}
                
                results = self.exporter.export_batch(
                    figs, base_path, formats, metadata
                )
                
                assert isinstance(results, dict)
                assert len(results) > 0
                
                # Dosyaların oluşturulduğunu kontrol et
                for i in range(3):
                    for format in formats:
                        file_path = base_path / f"figure_{i+1:02d}.{format}"
                        assert str(file_path) in results
                        assert results[str(file_path)] == True
                        assert file_path.exists()
        
        finally:
            for fig in figs:
                plt.close(fig)


class TestExportIntegration:
    """Dışa aktarma entegrasyon testleri"""
    
    def test_export_analysis_results(self):
        """Analiz sonuçlarını toplu dışa aktarma testi"""
        # Test figürleri oluştur
        figs = []
        for i in range(2):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title(f"Test Figure {i+1}")
            figs.append(fig)
        
        try:
            # Test verileri
            metrics_data = {
                'rms': np.random.randn(19),
                'mean': np.random.randn(19)
            }
            test_results = {
                'p_values': np.random.uniform(0, 1, 19),
                'statistics': np.random.randn(19)
            }
            metadata = {
                'analysis_info': {'timestamp': '2024-01-01T00:00:00'},
                'metrics': ['rms', 'mean']
            }
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir = Path(tmp_dir) / "output"
                formats = ['svg', 'png']
                
                success = export.export_analysis_results(
                    figs, metrics_data, test_results, output_dir, formats, metadata
                )
                
                assert success == True
                assert output_dir.exists()
                
                # Çıktı dosyalarını kontrol et
                svg_files = list(output_dir.glob("*.svg"))
                png_files = list(output_dir.glob("*.png"))
                json_files = list(output_dir.glob("*metadata.json"))
                
                assert len(svg_files) > 0
                assert len(png_files) > 0
                assert len(json_files) > 0
        
        finally:
            for fig in figs:
                plt.close(fig)


class TestExportEdgeCases:
    """Dışa aktarma kenar durumları testleri"""
    
    def test_export_figure_invalid_format(self):
        """Geçersiz format testi"""
        exporter = export.EEGExporter(verbose=False)
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([1, 2, 3], [1, 4, 2])
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.invalid', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            success = exporter.export_figure(fig, tmp_path, format='invalid')
            
            # Geçersiz format için başarısız olmalı
            assert success == False
            
        finally:
            plt.close(fig)
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_metadata_invalid_format(self):
        """Geçersiz metadata format testi"""
        exporter = export.EEGExporter(verbose=False)
        
        metadata = {'test': 'data'}
        
        with tempfile.NamedTemporaryFile(suffix='.invalid', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = exporter.export_metadata_file(metadata, tmp_path, format='invalid')
            
            # Geçersiz format için başarısız olmalı
            assert success == False
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_metrics_table_empty_data(self):
        """Boş veri ile metrik tablosu dışa aktarma testi"""
        exporter = export.EEGExporter(verbose=False)
        
        metrics_data = {}
        channel_names = []
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = exporter.export_metrics_table(
                metrics_data, channel_names, tmp_path, format='csv'
            )
            
            assert success == True
            assert Path(tmp_path).exists()
            
            # Boş CSV dosyasını kontrol et
            df = pd.read_csv(tmp_path)
            assert len(df) == 0
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
