"""
İstatistik modülü

Tek grup, iki grup ve çoklu karşılaştırma testleri.
FDR düzeltmesi ve cluster permutation testleri.
"""

from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon
from rich.console import Console

# Pingouin için
try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    console = Console()
    console.print("[yellow]pingouin kütüphanesi bulunamadı, bazı istatistiksel testler kullanılamayacak[/yellow]")

console = Console()


class EEGStatistics:
    """EEG istatistiksel analiz sınıfı"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = {}
        
    def single_group_test(
        self,
        data: np.ndarray,
        test_type: str = 'one_sample',
        baseline: Optional[float] = None,
        alternative: str = 'two-sided'
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Tek grup testi
        
        Parameters
        ----------
        data : np.ndarray
            Veri (kanallar x örnekler)
        test_type : str
            Test türü ('one_sample', 'z_score')
        baseline : float, optional
            Baseline değeri
        alternative : str
            Alternatif hipotez
            
        Returns
        -------
        dict
            Test sonuçları
        """
        if self.verbose:
            console.print("[blue]Tek grup testi hesaplanıyor...[/blue]")
            
        results = {}
        
        if test_type == 'one_sample':
            if baseline is None:
                baseline = 0.0
                
            # Tek örneklem t-testi
            t_stats, p_values = stats.ttest_1samp(data, baseline, axis=1)
            
            results = {
                'test_type': 'one_sample_ttest',
                'baseline': baseline,
                't_statistics': t_stats,
                'p_values': p_values,
                'alternative': alternative
            }
            
        elif test_type == 'z_score':
            # Z-score normalizasyonu
            z_scores = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
            
            results = {
                'test_type': 'z_score',
                'z_scores': z_scores,
                'mean': np.mean(data, axis=1),
                'std': np.std(data, axis=1)
            }
            
        else:
            raise ValueError(f"Desteklenmeyen tek grup testi: {test_type}")
            
        return results
    
    def two_group_test(
        self,
        group1_data: np.ndarray,
        group2_data: np.ndarray,
        test_type: str = 'ttest',
        paired: bool = False,
        alternative: str = 'two-sided',
        equal_var: bool = True
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        İki grup testi
        
        Parameters
        ----------
        group1_data : np.ndarray
            Birinci grup verisi
        group2_data : np.ndarray
            İkinci grup verisi
        test_type : str
            Test türü ('ttest', 'mannwhitney', 'wilcoxon')
        paired : bool
            Eşleşik test
        alternative : str
            Alternatif hipotez
        equal_var : bool
            Eşit varyans varsayımı
            
        Returns
        -------
        dict
            Test sonuçları
        """
        if self.verbose:
            console.print("[blue]İki grup testi hesaplanıyor...[/blue]")
            
        # Veri boyutlarını kontrol et
        if group1_data.shape[0] != group2_data.shape[0]:
            raise ValueError("Grup verileri aynı kanal sayısına sahip olmalı")
            
        n_channels = group1_data.shape[0]
        
        if test_type == 'ttest':
            if paired:
                # Eşleşik t-testi
                t_stats, p_values = ttest_rel(group1_data, group2_data, axis=1)
                test_name = 'paired_ttest'
            else:
                # Bağımsız t-testi
                t_stats, p_values = ttest_ind(group1_data, group2_data, axis=1, equal_var=equal_var)
                test_name = 'independent_ttest'
                
        elif test_type == 'mannwhitney':
            if paired:
                # Wilcoxon signed-rank test
                t_stats = np.zeros(n_channels)
                p_values = np.zeros(n_channels)
                
                for i in range(n_channels):
                    try:
                        stat, p_val = wilcoxon(group1_data[i, :], group2_data[i, :], alternative=alternative)
                        t_stats[i] = stat
                        p_values[i] = p_val
                    except ValueError:
                        t_stats[i] = np.nan
                        p_values[i] = np.nan
                        
                test_name = 'wilcoxon_signed_rank'
            else:
                # Mann-Whitney U test
                t_stats = np.zeros(n_channels)
                p_values = np.zeros(n_channels)
                
                for i in range(n_channels):
                    try:
                        stat, p_val = mannwhitneyu(group1_data[i, :], group2_data[i, :], alternative=alternative)
                        t_stats[i] = stat
                        p_values[i] = p_val
                    except ValueError:
                        t_stats[i] = np.nan
                        p_values[i] = np.nan
                        
                test_name = 'mannwhitney_u'
                
        else:
            raise ValueError(f"Desteklenmeyen iki grup testi: {test_type}")
            
        # Etki büyüklüğü hesapla
        effect_sizes = self._calculate_effect_size(group1_data, group2_data, paired=paired)
        
        results = {
            'test_type': test_name,
            'paired': paired,
            'alternative': alternative,
            'statistics': t_stats,
            'p_values': p_values,
            'effect_sizes': effect_sizes,
            'n_group1': group1_data.shape[1],
            'n_group2': group2_data.shape[1]
        }
        
        return results
    
    def multiple_comparison_correction(
        self,
        p_values: np.ndarray,
        method: str = 'fdr',
        alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Çoklu karşılaştırma düzeltmesi
        
        Parameters
        ----------
        p_values : np.ndarray
            P-değerleri
        method : str
            Düzeltme metodu ('fdr', 'bonferroni', 'holm')
        alpha : float
            Anlamlılık düzeyi
            
        Returns
        -------
        dict
            Düzeltme sonuçları
        """
        if self.verbose:
            console.print(f"[blue]Çoklu karşılaştırma düzeltmesi: {method}[/blue]")
            
        if method == 'fdr':
            # Benjamini-Hochberg FDR
            from statsmodels.stats.multitest import multipletests
            rejected, p_corrected, _, _ = multipletests(
                p_values, alpha=alpha, method='fdr_bh'
            )
            
        elif method == 'bonferroni':
            # Bonferroni düzeltmesi
            from statsmodels.stats.multitest import multipletests
            rejected, p_corrected, _, _ = multipletests(
                p_values, alpha=alpha, method='bonferroni'
            )
            
        elif method == 'holm':
            # Holm düzeltmesi
            from statsmodels.stats.multitest import multipletests
            rejected, p_corrected, _, _ = multipletests(
                p_values, alpha=alpha, method='holm'
            )
            
        else:
            raise ValueError(f"Desteklenmeyen düzeltme metodu: {method}")
            
        results = {
            'method': method,
            'alpha': alpha,
            'original_p_values': p_values,
            'corrected_p_values': p_corrected,
            'significant': rejected,
            'n_significant': np.sum(rejected)
        }
        
        if self.verbose:
            console.print(f"[green]{method} düzeltmesi: {results['n_significant']} anlamlı kanal[/green]")
            
        return results
    
    def cluster_permutation_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        adjacency: Optional[np.ndarray] = None,
        n_permutations: int = 1000,
        threshold: float = 0.05,
        tail: int = 0
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Cluster permutation testi
        
        Parameters
        ----------
        data1 : np.ndarray
            Birinci grup verisi
        data2 : np.ndarray
            İkinci grup verisi
        adjacency : np.ndarray, optional
            Kanal komşuluk matrisi
        n_permutations : int
            Permütasyon sayısı
        threshold : float
            Cluster eşiği
        tail : int
            Kuyruk türü (0: iki kuyruk, 1: sağ kuyruk, -1: sol kuyruk)
            
        Returns
        -------
        dict
            Cluster test sonuçları
        """
        if self.verbose:
            console.print("[blue]Cluster permutation testi hesaplanıyor...[/blue]")
            
        try:
            # MNE cluster permutation test
            from mne.stats import permutation_cluster_test
            
            # Veriyi MNE formatına çevir
            X = [data1.T, data2.T]  # (n_epochs, n_channels)
            
            # Cluster test
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                X,
                adjacency=adjacency,
                n_permutations=n_permutations,
                threshold=threshold,
                tail=tail,
                verbose=False
            )
            
            results = {
                'test_type': 'cluster_permutation',
                'T_obs': T_obs,
                'clusters': clusters,
                'cluster_p_values': cluster_p_values,
                'n_permutations': n_permutations,
                'threshold': threshold,
                'n_significant_clusters': np.sum(cluster_p_values < 0.05)
            }
            
            if self.verbose:
                console.print(f"[green]Cluster test: {results['n_significant_clusters']} anlamlı cluster[/green]")
                
        except ImportError:
            console.print("[yellow]MNE cluster test kullanılamıyor, basit permutation test uygulanıyor[/yellow]")
            results = self._simple_permutation_test(data1, data2, n_permutations)
            
        return results
    
    def _simple_permutation_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        n_permutations: int = 1000
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Basit permutation test"""
        n_channels = data1.shape[0]
        
        # Gözlemlenen istatistik
        observed_diff = np.mean(data1, axis=1) - np.mean(data2, axis=1)
        
        # Permütasyon testi
        all_data = np.concatenate([data1, data2], axis=1)
        n1 = data1.shape[1]
        n2 = data2.shape[1]
        
        permuted_diffs = []
        for _ in range(n_permutations):
            # Veriyi karıştır
            permuted_data = np.random.permutation(all_data.T).T
            perm_data1 = permuted_data[:, :n1]
            perm_data2 = permuted_data[:, n1:]
            
            perm_diff = np.mean(perm_data1, axis=1) - np.mean(perm_data2, axis=1)
            permuted_diffs.append(perm_diff)
            
        permuted_diffs = np.array(permuted_diffs)
        
        # P-değerleri hesapla
        p_values = np.zeros(n_channels)
        for i in range(n_channels):
            p_values[i] = np.mean(np.abs(permuted_diffs[:, i]) >= np.abs(observed_diff[i]))
            
        return {
            'test_type': 'simple_permutation',
            'observed_diff': observed_diff,
            'p_values': p_values,
            'n_permutations': n_permutations
        }
    
    def _calculate_effect_size(
        self,
        group1_data: np.ndarray,
        group2_data: np.ndarray,
        paired: bool = False
    ) -> np.ndarray:
        """Etki büyüklüğü hesapla"""
        n_channels = group1_data.shape[0]
        effect_sizes = np.zeros(n_channels)
        
        for i in range(n_channels):
            if paired:
                # Cohen's d (eşleşik)
                diff = group1_data[i, :] - group2_data[i, :]
                pooled_std = np.std(diff)
                if pooled_std > 0:
                    effect_sizes[i] = np.mean(diff) / pooled_std
            else:
                # Cohen's d (bağımsız)
                pooled_std = np.sqrt(
                    ((group1_data.shape[1] - 1) * np.var(group1_data[i, :]) +
                     (group2_data.shape[1] - 1) * np.var(group2_data[i, :])) / \
                    (group1_data.shape[1] + group2_data.shape[1] - 2)
                )
                if pooled_std > 0:
                    effect_sizes[i] = (np.mean(group1_data[i, :]) - np.mean(group2_data[i, :])) / pooled_std
                    
        return effect_sizes
    
    def parametric_assumptions_test(
        self,
        data: np.ndarray,
        group_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Union[bool, np.ndarray]]:
        """
        Parametrik varsayımları test et
        
        Parameters
        ----------
        data : np.ndarray
            Veri
        group_labels : np.ndarray, optional
            Grup etiketleri
            
        Returns
        -------
        dict
            Varsayım test sonuçları
        """
        if self.verbose:
            console.print("[blue]Parametrik varsayımlar test ediliyor...[/blue]")
            
        results = {}
        
        # Normallik testi (Shapiro-Wilk)
        n_channels = data.shape[0]
        normality_p = np.zeros(n_channels)
        
        for i in range(n_channels):
            try:
                _, p_val = stats.shapiro(data[i, :])
                normality_p[i] = p_val
            except:
                normality_p[i] = np.nan
                
        results['normality_p_values'] = normality_p
        results['normality_passed'] = normality_p > 0.05
        
        # Homojenlik testi (Levene)
        if group_labels is not None:
            try:
                levene_stat, levene_p = stats.levene(*[data[i, group_labels == g] for g in np.unique(group_labels)])
                results['levene_p_value'] = levene_p
                results['homogeneity_passed'] = levene_p > 0.05
            except:
                results['levene_p_value'] = np.nan
                results['homogeneity_passed'] = False
        else:
            results['levene_p_value'] = np.nan
            results['homogeneity_passed'] = True
            
        return results
    
    def get_statistical_summary(
        self,
        test_results: Dict,
        channel_names: List[str]
    ) -> pd.DataFrame:
        """
        İstatistiksel özet tablosu oluştur
        
        Parameters
        ----------
        test_results : dict
            Test sonuçları
        channel_names : list
            Kanal adları
            
        Returns
        -------
        pd.DataFrame
            Özet tablosu
        """
        summary_data = {'channel': channel_names}
        
        # Test türüne göre veri ekle
        if 'p_values' in test_results:
            summary_data['p_value'] = test_results['p_values']
            
        if 'statistics' in test_results:
            summary_data['statistic'] = test_results['statistics']
            
        if 'effect_sizes' in test_results:
            summary_data['effect_size'] = test_results['effect_sizes']
            
        if 'corrected_p_values' in test_results:
            summary_data['corrected_p_value'] = test_results['corrected_p_values']
            summary_data['significant'] = test_results['significant']
            
        return pd.DataFrame(summary_data)


def perform_statistical_analysis(
    data_dict: Dict[str, np.ndarray],
    analysis_type: str = 'two_group',
    test_type: str = 'ttest',
    correction_method: str = 'fdr',
    alpha: float = 0.05,
    **kwargs
) -> Dict[str, Union[Dict, pd.DataFrame]]:
    """
    İstatistiksel analiz ana fonksiyonu
    
    Parameters
    ----------
    data_dict : dict
        Veri sözlüğü (grup adları -> veri)
    analysis_type : str
        Analiz türü
    test_type : str
        Test türü
    correction_method : str
        Düzeltme metodu
    alpha : float
        Anlamlılık düzeyi
    **kwargs
        Diğer parametreler
        
    Returns
    -------
    dict
        Analiz sonuçları
    """
    stats_analyzer = EEGStatistics()
    
    if analysis_type == 'single_group':
        # Tek grup analizi
        group_data = list(data_dict.values())[0]
        results = stats_analyzer.single_group_test(group_data, test_type=test_type, **kwargs)
        
    elif analysis_type == 'two_group':
        # İki grup analizi
        group_names = list(data_dict.keys())
        if len(group_names) != 2:
            raise ValueError("İki grup analizi için tam olarak 2 grup gerekli")
            
        group1_data = data_dict[group_names[0]]
        group2_data = data_dict[group_names[1]]
        
        results = stats_analyzer.two_group_test(
            group1_data, group2_data, test_type=test_type, **kwargs
        )
        
        # Çoklu karşılaştırma düzeltmesi
        if 'p_values' in results:
            correction_results = stats_analyzer.multiple_comparison_correction(
                results['p_values'], method=correction_method, alpha=alpha
            )
            results.update(correction_results)
            
    else:
        raise ValueError(f"Desteklenmeyen analiz türü: {analysis_type}")
        
    return results
