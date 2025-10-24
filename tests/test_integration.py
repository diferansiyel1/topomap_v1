"""
Entegrasyon testleri

Tüm modüllerin birlikte çalışmasını test eder.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from eeg_topomap_lab import io, preproc, metrics, bipolar, stats, viz, export


@pytest.mark.integration
class TestFullAnalysisPipeline:
    """Tam analiz pipeline testi"""
    
    def test_complete_analysis_pipeline(self, sample_eeg_data, temp_directory):
        """Tam analiz pipeline testi"""
        raw = sample_eeg_data
        
        # 1. Ön işleme
        preprocessor = preproc.EEGPreprocessor(verbose=False)
        processed_raw = preprocessor.apply_filters(raw, l_freq=1.0, h_freq=40.0)
        processed_raw = preprocessor.apply_reference(processed_raw, ref_type='average')
        
        # 2. Segmentleme
        segments = {'preiktal': [2, 5], 'interiktal': [7, 10]}
        segmented_data = preprocessor.segment_data(processed_raw, segments)
        
        # 3. Metrik hesaplama
        metrics_calculator = metrics.EEGMetrics(verbose=False)
        all_metrics = {}
        
        for seg_name, seg_raw in segmented_data.items():
            # Zaman domeni metrikleri
            time_metrics = metrics_calculator.calculate_time_domain_metrics(
                seg_raw, ['rms', 'mean']
            )
            
            # Frekans domeni metrikleri
            freq_metrics = metrics_calculator.calculate_frequency_domain_metrics(
                seg_raw, ['band_power'], 
                bands={'alpha': (8, 13), 'beta': (13, 30)}
            )
            
            # Metrikleri birleştir
            seg_metrics = {**time_metrics, **freq_metrics}
            all_metrics[seg_name] = seg_metrics
        
        # 4. İstatistiksel analiz
        stats_analyzer = stats.EEGStatistics(verbose=False)
        
        # İki grup karşılaştırması
        group1_data = all_metrics['preiktal']['rms']
        group2_data = all_metrics['interiktal']['rms']
        
        test_results = stats_analyzer.two_group_test(
            group1_data, group2_data, test_type='ttest', paired=False
        )
        
        # FDR düzeltmesi
        correction_results = stats_analyzer.multiple_comparison_correction(
            test_results['p_values'], method='fdr', alpha=0.05
        )
        test_results.update(correction_results)
        
        # 5. Görselleştirme
        visualizer = viz.EEGVisualizer(verbose=False)
        
        # Fark haritası
        fig = visualizer.plot_difference_topomap(
            all_metrics['preiktal']['rms'],
            all_metrics['interiktal']['rms'],
            raw.info,
            significance_mask=test_results.get('significant'),
            title="Preiktal vs Interiktal - RMS"
        )
        
        # 6. Dışa aktarma
        exporter = export.EEGExporter(verbose=False)
        
        # Figürü kaydet
        output_path = temp_directory / "analysis_result.svg"
        success = exporter.export_figure(fig, output_path, format='svg')
        
        assert success == True
        assert output_path.exists()
        
        # Metadata oluştur ve kaydet
        metadata = exporter.create_analysis_metadata(
            "test.edf", "standard_1020", segments, ['rms', 'alpha'],
            test_results, {'cmap': 'viridis'}, {'l_freq': 1.0, 'h_freq': 40.0}
        )
        
        metadata_path = temp_directory / "analysis_metadata.json"
        exporter.export_metadata_file(metadata, metadata_path)
        
        assert metadata_path.exists()
        
        # Metrik tablosunu kaydet
        metrics_table_path = temp_directory / "metrics_table.csv"
        exporter.export_metrics_table(
            all_metrics['preiktal'], raw.ch_names, metrics_table_path
        )
        
        assert metrics_table_path.exists()
        
        # İstatistiksel sonuçları kaydet
        stats_table_path = temp_directory / "statistical_results.csv"
        exporter.export_statistical_results(
            test_results, raw.ch_names, stats_table_path
        )
        
        assert stats_table_path.exists()


@pytest.mark.integration
class TestBipolarAnalysisPipeline:
    """Bipolar analiz pipeline testi"""
    
    def test_bipolar_analysis_pipeline(self, bipolar_eeg_data, temp_directory):
        """Bipolar analiz pipeline testi"""
        raw = bipolar_eeg_data
        
        # 1. Bipolar işleme
        processor = bipolar.BipolarProcessor(verbose=False)
        bipolar_pairs = processor.detect_bipolar_channels(raw)
        
        # Referans dönüşümü
        converted_raw = processor.convert_bipolar_to_reference(
            raw, ref_type='average'
        )
        
        # 2. Ön işleme
        preprocessor = preproc.EEGPreprocessor(verbose=False)
        processed_raw = preprocessor.apply_filters(converted_raw, l_freq=1.0, h_freq=40.0)
        
        # 3. Segmentleme
        segments = {'preiktal': [2, 5], 'interiktal': [7, 10]}
        segmented_data = preprocessor.segment_data(processed_raw, segments)
        
        # 4. Metrik hesaplama
        metrics_calculator = metrics.EEGMetrics(verbose=False)
        all_metrics = {}
        
        for seg_name, seg_raw in segmented_data.items():
            # DFA hesaplama
            dfa_metrics = metrics_calculator.calculate_nonlinear_metrics(
                seg_raw, ['dfa']
            )
            all_metrics[seg_name] = dfa_metrics
        
        # 5. Görselleştirme
        visualizer = viz.EEGVisualizer(verbose=False)
        
        # Çoklu panel topomap
        panel_data = {
            name: metrics['dfa'] for name, metrics in all_metrics.items()
        }
        
        fig = visualizer.plot_multi_panel_topomap(
            panel_data, converted_raw.info, rows=1, cols=2
        )
        
        # 6. Dışa aktarma
        exporter = export.EEGExporter(verbose=False)
        
        output_path = temp_directory / "bipolar_analysis.svg"
        success = exporter.export_figure(fig, output_path, format='svg')
        
        assert success == True
        assert output_path.exists()


@pytest.mark.integration
class TestCLIIntegration:
    """CLI entegrasyon testi"""
    
    def test_cli_analysis_workflow(self, mock_eeg_file, temp_directory):
        """CLI analiz workflow testi"""
        # Bu test CLI'nin temel işlevselliğini test eder
        # Gerçek CLI çağrısı yapmak yerine, CLI'nin kullandığı
        # fonksiyonları doğrudan test ederiz
        
        # 1. Veri yükleme
        loader = io.EEGDataLoader(verbose=False)
        raw = loader.load_data(mock_eeg_file, montage="standard_1020")
        
        assert raw is not None
        assert len(raw.ch_names) > 0
        
        # 2. Ön işleme
        preprocessor = preproc.EEGPreprocessor(verbose=False)
        processed_raw = preprocessor.apply_filters(raw, l_freq=1.0, h_freq=40.0)
        processed_raw = preprocessor.apply_reference(processed_raw, ref_type='average')
        
        # 3. Segmentleme
        segments = {'preiktal': [1, 3], 'interiktal': [4, 6]}
        segmented_data = preprocessor.segment_data(processed_raw, segments)
        
        # 4. Metrik hesaplama
        metrics_calculator = metrics.EEGMetrics(verbose=False)
        all_metrics = {}
        
        for seg_name, seg_raw in segmented_data.items():
            time_metrics = metrics_calculator.calculate_time_domain_metrics(
                seg_raw, ['rms', 'mean']
            )
            all_metrics[seg_name] = time_metrics
        
        # 5. İstatistiksel analiz
        stats_analyzer = stats.EEGStatistics(verbose=False)
        
        group1_data = all_metrics['preiktal']['rms']
        group2_data = all_metrics['interiktal']['rms']
        
        test_results = stats_analyzer.two_group_test(
            group1_data, group2_data, test_type='ttest', paired=False
        )
        
        # 6. Görselleştirme
        visualizer = viz.EEGVisualizer(verbose=False)
        
        fig = visualizer.plot_difference_topomap(
            all_metrics['preiktal']['rms'],
            all_metrics['interiktal']['rms'],
            raw.info,
            title="Preiktal vs Interiktal"
        )
        
        # 7. Dışa aktarma
        exporter = export.EEGExporter(verbose=False)
        
        output_path = temp_directory / "cli_test_result.svg"
        success = exporter.export_figure(fig, output_path, format='svg')
        
        assert success == True
        assert output_path.exists()


@pytest.mark.integration
class TestErrorHandling:
    """Hata yönetimi entegrasyon testi"""
    
    def test_invalid_input_handling(self):
        """Geçersiz girdi hata yönetimi testi"""
        # Var olmayan dosya
        loader = io.EEGDataLoader(verbose=False)
        
        with pytest.raises(FileNotFoundError):
            loader.load_data("nonexistent_file.edf")
    
    def test_invalid_metric_handling(self, sample_eeg_data):
        """Geçersiz metrik hata yönetimi testi"""
        raw = sample_eeg_data
        metrics_calculator = metrics.EEGMetrics(verbose=False)
        
        # Geçersiz metrik adı
        results = metrics_calculator.calculate_time_domain_metrics(
            raw, ['invalid_metric']
        )
        
        # Geçersiz metrik için boş sonuç döndürülmeli
        assert 'invalid_metric' not in results
    
    def test_insufficient_data_handling(self):
        """Yetersiz veri hata yönetimi testi"""
        # Çok kısa veri
        sfreq = 250
        duration = 0.1  # Çok kısa
        n_channels = 19
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # DFA hesaplama (yetersiz veri ile)
        metrics_calculator = metrics.EEGMetrics(verbose=False)
        results = metrics_calculator.calculate_nonlinear_metrics(
            raw, ['dfa']
        )
        
        # DFA hesaplanamayabilir, bu normal
        if 'dfa' in results:
            assert isinstance(results['dfa'], np.ndarray)


@pytest.mark.integration
class TestPerformance:
    """Performans entegrasyon testi"""
    
    @pytest.mark.slow
    def test_large_dataset_handling(self):
        """Büyük veri seti işleme testi"""
        # Büyük veri seti oluştur
        sfreq = 1000
        duration = 60  # 1 dakika
        n_channels = 64
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Ön işleme
        preprocessor = preproc.EEGPreprocessor(verbose=False)
        processed_raw = preprocessor.apply_filters(raw, l_freq=1.0, h_freq=40.0)
        
        # Segmentleme
        segments = {'segment1': [10, 20], 'segment2': [30, 40]}
        segmented_data = preprocessor.segment_data(processed_raw, segments)
        
        # Metrik hesaplama
        metrics_calculator = metrics.EEGMetrics(verbose=False)
        
        for seg_name, seg_raw in segmented_data.items():
            time_metrics = metrics_calculator.calculate_time_domain_metrics(
                seg_raw, ['rms', 'mean']
            )
            
            assert 'rms' in time_metrics
            assert 'mean' in time_metrics
            assert time_metrics['rms'].shape[0] == n_channels
    
    def test_memory_efficient_processing(self):
        """Bellek verimli işleme testi"""
        # Orta boyutlu veri seti
        sfreq = 500
        duration = 30
        n_channels = 32
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Bellek verimli işleme
        preprocessor = preproc.EEGPreprocessor(verbose=False)
        
        # Segmentleme (bellek verimli)
        segments = {'segment1': [5, 10], 'segment2': [15, 20]}
        segmented_data = preprocessor.segment_data(raw, segments)
        
        # Her segmenti ayrı ayrı işle
        for seg_name, seg_raw in segmented_data.items():
            # Filtre uygula
            filtered_raw = preprocessor.apply_filters(seg_raw, l_freq=1.0, h_freq=40.0)
            
            # Metrik hesapla
            metrics_calculator = metrics.EEGMetrics(verbose=False)
            time_metrics = metrics_calculator.calculate_time_domain_metrics(
                filtered_raw, ['rms']
            )
            
            assert 'rms' in time_metrics
            assert time_metrics['rms'].shape[0] == n_channels
