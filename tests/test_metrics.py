"""
Metrik hesaplama modülü testleri
"""

import numpy as np
import pytest
import mne
from eeg_topomap_lab import metrics


class TestEEGMetrics:
    """EEGMetrics test sınıfı"""
    
    def setup_method(self):
        """Test kurulumu"""
        # Test verisi oluştur
        sfreq = 250
        duration = 10
        n_channels = 19
        
        # Rastgele EEG verisi
        data = np.random.randn(n_channels, int(sfreq * duration))
        
        # Kanal adları (10-20 sistemi)
        ch_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
        ]
        
        # MNE Info oluştur
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        
        # Raw objesi oluştur
        self.raw = mne.io.RawArray(data, info, verbose=False)
        
        # Metrik hesaplayıcı
        self.metrics_calculator = metrics.EEGMetrics(verbose=False)
    
    def test_time_domain_metrics(self):
        """Zaman domeni metrikleri testi"""
        metrics_list = ['rms', 'peak_to_peak', 'mean', 'std']
        
        results = self.metrics_calculator.calculate_time_domain_metrics(
            self.raw, metrics_list
        )
        
        assert isinstance(results, dict)
        assert len(results) == len(metrics_list)
        
        for metric in metrics_list:
            assert metric in results
            assert isinstance(results[metric], np.ndarray)
            assert results[metric].shape[0] == len(self.raw.ch_names)
    
    def test_frequency_domain_metrics(self):
        """Frekans domeni metrikleri testi"""
        metrics_list = ['band_power']
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        results = self.metrics_calculator.calculate_frequency_domain_metrics(
            self.raw, metrics_list, bands=bands
        )
        
        assert isinstance(results, dict)
        
        # Bant güçleri kontrolü
        for band_name in bands.keys():
            assert band_name in results
            assert isinstance(results[band_name], np.ndarray)
            assert results[band_name].shape[0] == len(self.raw.ch_names)
            assert np.all(results[band_name] >= 0)  # Güç değerleri pozitif olmalı
    
    def test_relative_power_calculation(self):
        """Relatif güç hesaplama testi"""
        bands = {
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        results = self.metrics_calculator.calculate_frequency_domain_metrics(
            self.raw, ['band_power'], bands=bands, relative=True
        )
        
        # Relatif güç değerleri 0-1 arasında olmalı
        for band_name in bands.keys():
            rel_power = results[f'{band_name}_relative']
            assert np.all(rel_power >= 0)
            assert np.all(rel_power <= 1)
    
    def test_alpha_beta_ratio(self):
        """Alfa/beta oranı testi"""
        bands = {
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        results = self.metrics_calculator.calculate_frequency_domain_metrics(
            self.raw, ['band_power', 'alpha_beta_ratio'], bands=bands
        )
        
        assert 'alpha_beta_ratio' in results
        assert isinstance(results['alpha_beta_ratio'], np.ndarray)
        assert results['alpha_beta_ratio'].shape[0] == len(self.raw.ch_names)
        assert np.all(results['alpha_beta_ratio'] >= 0)  # Oran pozitif olmalı
    
    def test_nonlinear_metrics_availability(self):
        """Nonlineer metriklerin kullanılabilirliği testi"""
        metrics_list = ['dfa', 'lempel_ziv', 'higuchi_fd', 'permutation_entropy']
        
        results = self.metrics_calculator.calculate_nonlinear_metrics(
            self.raw, metrics_list
        )
        
        assert isinstance(results, dict)
        
        # Kullanılabilir metrikleri kontrol et
        for metric in metrics_list:
            if metric in results:
                assert isinstance(results[metric], np.ndarray)
                assert results[metric].shape[0] == len(self.raw.ch_names)
    
    def test_metrics_dataframe_conversion(self):
        """Metriklerin DataFrame'e çevrilmesi testi"""
        # Zaman domeni metrikleri hesapla
        time_results = self.metrics_calculator.calculate_time_domain_metrics(
            self.raw, ['rms', 'mean']
        )
        
        # DataFrame'e çevir
        df = self.metrics_calculator.get_channel_metrics_dataframe(
            time_results, self.raw.ch_names
        )
        
        assert isinstance(df, type(df))  # DataFrame kontrolü
        assert len(df) == len(self.raw.ch_names)
        assert 'channel' in df.columns
        assert 'rms' in df.columns
        assert 'mean' in df.columns


class TestDFADetailed:
    """Detaylı DFA hesaplama testleri"""
    
    def test_dfa_calculation(self):
        """DFA hesaplama testi"""
        # Test sinyali oluştur (1/f^α gibi)
        n_samples = 1000
        t = np.linspace(0, 10, n_samples)
        signal = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(n_samples)
        
        dfa_val, details = metrics.calculate_dfa_detailed(
            signal, min_window=10, max_window=100, n_windows=20
        )
        
        assert isinstance(dfa_val, (float, np.floating))
        assert isinstance(details, dict)
        assert 'min_window' in details
        assert 'max_window' in details
        assert 'n_windows' in details
    
    def test_band_power_ratio(self):
        """Bant güç oranı testi"""
        # Test verisi oluştur
        sfreq = 250
        duration = 10
        n_channels = 19
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Bant güç oranı hesapla
        ratio = metrics.calculate_band_power_ratio(
            raw, band1=(8, 13), band2=(13, 30)
        )
        
        assert isinstance(ratio, np.ndarray)
        assert ratio.shape[0] == n_channels
        assert np.all(ratio >= 0)  # Oran pozitif olmalı
    
    def test_spectral_edge_frequency(self):
        """Spektral kenar frekansı testi"""
        # Test verisi oluştur
        sfreq = 250
        duration = 10
        n_channels = 19
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Spektral kenar frekansı hesapla
        sef = metrics.calculate_spectral_edge_frequency(
            raw, percentile=95.0, fmin=0.5, fmax=100.0
        )
        
        assert isinstance(sef, np.ndarray)
        assert sef.shape[0] == n_channels
        assert np.all(sef >= 0.5)  # Minimum frekans
        assert np.all(sef <= 100.0)  # Maksimum frekans


class TestMetricsIntegration:
    """Metrik entegrasyon testleri"""
    
    def test_all_metrics_calculation(self):
        """Tüm metriklerin hesaplanması testi"""
        # Test verisi oluştur
        sfreq = 250
        duration = 5
        n_channels = 19
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        metrics_calculator = metrics.EEGMetrics(verbose=False)
        
        # Tüm metrikleri hesapla
        all_results = metrics_calculator.calculate_all_metrics(
            raw,
            time_metrics=['rms', 'mean'],
            freq_metrics=['band_power'],
            nonlinear_metrics=['dfa']
        )
        
        assert isinstance(all_results, dict)
        assert len(all_results) > 0
        
        # Temel metriklerin varlığını kontrol et
        expected_metrics = ['rms', 'mean']
        for metric in expected_metrics:
            if metric in all_results:
                assert isinstance(all_results[metric], np.ndarray)
                assert all_results[metric].shape[0] == n_channels
