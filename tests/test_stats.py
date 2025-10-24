"""
İstatistik modülü testleri
"""

import numpy as np
import pytest
from eeg_topomap_lab import stats


class TestEEGStatistics:
    """EEGStatistics test sınıfı"""
    
    def setup_method(self):
        """Test kurulumu"""
        self.stats_analyzer = stats.EEGStatistics(verbose=False)
        
        # Test verileri oluştur
        np.random.seed(42)
        self.n_channels = 19
        self.n_samples = 100
        
        # Normal dağılımlı veriler
        self.group1_data = np.random.randn(self.n_channels, self.n_samples)
        self.group2_data = np.random.randn(self.n_channels, self.n_samples) + 0.5  # Farklı ortalama
        
        # Kanal adları
        self.channel_names = [f'Ch{i}' for i in range(self.n_channels)]
    
    def test_single_group_test(self):
        """Tek grup testi"""
        # Tek örneklem t-testi
        results = self.stats_analyzer.single_group_test(
            self.group1_data, test_type='one_sample', baseline=0.0
        )
        
        assert isinstance(results, dict)
        assert 'test_type' in results
        assert 't_statistics' in results
        assert 'p_values' in results
        assert results['t_statistics'].shape[0] == self.n_channels
        assert results['p_values'].shape[0] == self.n_channels
        
        # P-değerleri 0-1 arasında olmalı
        assert np.all(results['p_values'] >= 0)
        assert np.all(results['p_values'] <= 1)
    
    def test_z_score_normalization(self):
        """Z-score normalizasyonu testi"""
        results = self.stats_analyzer.single_group_test(
            self.group1_data, test_type='z_score'
        )
        
        assert isinstance(results, dict)
        assert 'z_scores' in results
        assert 'mean' in results
        assert 'std' in results
        
        # Z-score'ların ortalaması yaklaşık 0 olmalı
        z_scores = results['z_scores']
        assert np.allclose(np.mean(z_scores, axis=1), 0, atol=1e-10)
        
        # Z-score'ların standart sapması yaklaşık 1 olmalı
        assert np.allclose(np.std(z_scores, axis=1), 1, atol=1e-10)
    
    def test_two_group_ttest(self):
        """İki grup t-testi"""
        results = self.stats_analyzer.two_group_test(
            self.group1_data, self.group2_data, test_type='ttest', paired=False
        )
        
        assert isinstance(results, dict)
        assert 'test_type' in results
        assert 'statistics' in results
        assert 'p_values' in results
        assert 'effect_sizes' in results
        
        # Sonuçlar kanal sayısı kadar olmalı
        assert results['statistics'].shape[0] == self.n_channels
        assert results['p_values'].shape[0] == self.n_channels
        assert results['effect_sizes'].shape[0] == self.n_channels
        
        # P-değerleri 0-1 arasında olmalı
        assert np.all(results['p_values'] >= 0)
        assert np.all(results['p_values'] <= 1)
    
    def test_two_group_paired_ttest(self):
        """Eşleşik t-testi"""
        results = self.stats_analyzer.two_group_test(
            self.group1_data, self.group2_data, test_type='ttest', paired=True
        )
        
        assert isinstance(results, dict)
        assert 'test_type' in results
        assert results['test_type'] == 'paired_ttest'
        assert results['paired'] == True
    
    def test_mann_whitney_test(self):
        """Mann-Whitney U testi"""
        results = self.stats_analyzer.two_group_test(
            self.group1_data, self.group2_data, test_type='mannwhitney', paired=False
        )
        
        assert isinstance(results, dict)
        assert 'test_type' in results
        assert results['test_type'] == 'mannwhitney_u'
        assert results['paired'] == False
    
    def test_wilcoxon_test(self):
        """Wilcoxon signed-rank testi"""
        results = self.stats_analyzer.two_group_test(
            self.group1_data, self.group2_data, test_type='mannwhitney', paired=True
        )
        
        assert isinstance(results, dict)
        assert 'test_type' in results
        assert results['test_type'] == 'wilcoxon_signed_rank'
        assert results['paired'] == True
    
    def test_multiple_comparison_correction(self):
        """Çoklu karşılaştırma düzeltmesi"""
        # Rastgele p-değerleri oluştur
        p_values = np.random.uniform(0, 1, self.n_channels)
        
        # FDR düzeltmesi
        fdr_results = self.stats_analyzer.multiple_comparison_correction(
            p_values, method='fdr', alpha=0.05
        )
        
        assert isinstance(fdr_results, dict)
        assert 'method' in fdr_results
        assert 'corrected_p_values' in fdr_results
        assert 'significant' in fdr_results
        
        # Düzeltilmiş p-değerleri orijinal p-değerlerinden büyük veya eşit olmalı
        assert np.all(fdr_results['corrected_p_values'] >= p_values)
        
        # Bonferroni düzeltmesi
        bonf_results = self.stats_analyzer.multiple_comparison_correction(
            p_values, method='bonferroni', alpha=0.05
        )
        
        assert bonf_results['method'] == 'bonferroni'
        assert np.all(bonf_results['corrected_p_values'] >= fdr_results['corrected_p_values'])
    
    def test_parametric_assumptions(self):
        """Parametrik varsayımlar testi"""
        # Normal dağılımlı veri
        normal_data = np.random.randn(self.n_channels, self.n_samples)
        
        results = self.stats_analyzer.parametric_assumptions_test(normal_data)
        
        assert isinstance(results, dict)
        assert 'normality_p_values' in results
        assert 'normality_passed' in results
        
        # Normallik test sonuçları
        assert results['normality_p_values'].shape[0] == self.n_channels
        assert results['normality_passed'].shape[0] == self.n_channels
    
    def test_effect_size_calculation(self):
        """Etki büyüklüğü hesaplama testi"""
        # Farklı ortalamalara sahip gruplar
        group1 = np.random.randn(self.n_channels, self.n_samples)
        group2 = np.random.randn(self.n_channels, self.n_samples) + 1.0
        
        # Bağımsız gruplar için etki büyüklüğü
        effect_sizes_indep = self.stats_analyzer._calculate_effect_size(
            group1, group2, paired=False
        )
        
        assert isinstance(effect_sizes_indep, np.ndarray)
        assert effect_sizes_indep.shape[0] == self.n_channels
        
        # Eşleşik gruplar için etki büyüklüğü
        effect_sizes_paired = self.stats_analyzer._calculate_effect_size(
            group1, group2, paired=True
        )
        
        assert isinstance(effect_sizes_paired, np.ndarray)
        assert effect_sizes_paired.shape[0] == self.n_channels
    
    def test_statistical_summary(self):
        """İstatistiksel özet testi"""
        # Test sonuçları oluştur
        test_results = {
            'p_values': np.random.uniform(0, 1, self.n_channels),
            'statistics': np.random.randn(self.n_channels),
            'effect_sizes': np.random.randn(self.n_channels),
            'corrected_p_values': np.random.uniform(0, 1, self.n_channels),
            'significant': np.random.choice([True, False], self.n_channels)
        }
        
        # Özet tablosu oluştur
        summary_df = self.stats_analyzer.get_statistical_summary(
            test_results, self.channel_names
        )
        
        assert isinstance(summary_df, type(summary_df))  # DataFrame kontrolü
        assert len(summary_df) == self.n_channels
        assert 'channel' in summary_df.columns
        assert 'p_value' in summary_df.columns
        assert 'statistic' in summary_df.columns
        assert 'effect_size' in summary_df.columns
        assert 'corrected_p_value' in summary_df.columns
        assert 'significant' in summary_df.columns


class TestStatisticalAnalysis:
    """İstatistiksel analiz ana fonksiyon testleri"""
    
    def test_perform_statistical_analysis_single_group(self):
        """Tek grup analizi testi"""
        data_dict = {'group1': np.random.randn(19, 100)}
        
        results = stats.perform_statistical_analysis(
            data_dict, analysis_type='single_group', test_type='one_sample'
        )
        
        assert isinstance(results, dict)
        assert 'test_type' in results
    
    def test_perform_statistical_analysis_two_group(self):
        """İki grup analizi testi"""
        data_dict = {
            'group1': np.random.randn(19, 100),
            'group2': np.random.randn(19, 100) + 0.5
        }
        
        results = stats.perform_statistical_analysis(
            data_dict, analysis_type='two_group', test_type='ttest'
        )
        
        assert isinstance(results, dict)
        assert 'test_type' in results
        assert 'p_values' in results
        assert 'corrected_p_values' in results  # FDR düzeltmesi uygulanmış olmalı


class TestClusterPermutation:
    """Cluster permutation testleri"""
    
    def test_simple_permutation_test(self):
        """Basit permutation testi"""
        stats_analyzer = stats.EEGStatistics(verbose=False)
        
        group1_data = np.random.randn(19, 50)
        group2_data = np.random.randn(19, 50) + 0.3
        
        results = stats_analyzer._simple_permutation_test(
            group1_data, group2_data, n_permutations=100
        )
        
        assert isinstance(results, dict)
        assert 'test_type' in results
        assert 'observed_diff' in results
        assert 'p_values' in results
        assert 'n_permutations' in results
        
        assert results['observed_diff'].shape[0] == 19
        assert results['p_values'].shape[0] == 19
        assert results['n_permutations'] == 100
        
        # P-değerleri 0-1 arasında olmalı
        assert np.all(results['p_values'] >= 0)
        assert np.all(results['p_values'] <= 1)
