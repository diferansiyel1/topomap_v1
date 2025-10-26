"""
Görselleştirme modülü

Topolojik harita çizimi, panel grid düzeni, istatistiksel maskeleme.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from rich.console import Console

console = Console()


class EEGVisualizer:
    """EEG görselleştirme sınıfı"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.fig = None
        self.axes = None
        
    def plot_topomap(
        self,
        values: np.ndarray,
        info: mne.Info,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = 'viridis',
        contours: int = 6,
        show_names: bool = False,
        mask: Optional[np.ndarray] = None,
        mask_params: Optional[Dict] = None,
        title: Optional[str] = None,
        axes: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Topolojik harita çiz - optimize edilmiş versiyon
        
        Parameters
        ----------
        values : np.ndarray
            Kanal değerleri
        info : mne.Info
            Kanal bilgileri
        vmin : float, optional
            Minimum değer
        vmax : float, optional
            Maksimum değer
        cmap : str
            Renk haritası
        contours : int
            Kontur sayısı
        show_names : bool
            Kanal adlarını göster
        mask : np.ndarray, optional
            Maskeleme array'i
        mask_params : dict, optional
            Maskeleme parametreleri
        title : str, optional
            Başlık
        axes : plt.Axes, optional
            Matplotlib ekseni
        **kwargs
            Diğer parametreler
            
        Returns
        -------
        plt.Figure
            Çizilen figür
        """
        if self.verbose:
            console.print("[blue]Topolojik harita çiziliyor...[/blue]")
            
        # Varsayılan maskeleme parametreleri - iyileştirilmiş
        if mask_params is None:
            mask_params = {
                'marker': 'o',
                'markerfacecolor': 'w',
                'markeredgecolor': 'k',
                'markersize': 6,
                'linewidth': 1
            }
            
        # Figür boyutunu optimize et - daha büyük boyut
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(12, 10))
        else:
            fig = axes.figure
            
        # MNE topomap çiz - klasik yuvarlak format
        im, _ = mne.viz.plot_topomap(
            values,
            info,
            cmap=cmap,
            contours=contours,
            mask=mask,
            mask_params=mask_params,
            axes=axes,
            show=False,
            outlines='head',  # Show classic head outline with ears/nose
            sphere=0.15,       # Manually set sphere radius for larger head outline
            extrapolate='head',  # Only interpolate within head
            border='mean',    # Add border for smoother edges
            res=128,  # Higher resolution for better quality
            **kwargs
        )
        
        # Renk skalasını manuel olarak ayarla - iyileştirilmiş
        if vmin is not None and vmax is not None:
            im.set_clim(vmin=vmin, vmax=vmax)
        else:
            # Otomatik ölçekleme - daha iyi dağılım için
            vmin_auto = np.percentile(values, 5)
            vmax_auto = np.percentile(values, 95)
            im.set_clim(vmin=vmin_auto, vmax=vmax_auto)
        
        # Başlık ekle - iyileştirilmiş
        if title:
            axes.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
        # Renk çubuğu ekle - iyileştirilmiş
        if im is not None:
            cbar = plt.colorbar(im, ax=axes, shrink=0.8, aspect=20)
            cbar.set_label('Değer', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
        # Eksen ayarları - iyileştirilmiş
        axes.set_aspect('equal')
        axes.axis('off')
        
        return fig
    
    def plot_multi_panel_topomap(
        self,
        data_dict: Dict[str, np.ndarray],
        info: mne.Info,
        rows: int = 2,
        cols: int = 2,
        figsize: Tuple[float, float] = (12, 8),
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = 'viridis',
        contours: int = 6,
        show_names: bool = False,
        masks: Optional[Dict[str, np.ndarray]] = None,
        titles: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Çoklu panel topolojik harita çiz
        
        Parameters
        ----------
        data_dict : dict
            Panel verileri (panel_adı -> kanal_değerleri)
        info : mne.Info
            Kanal bilgileri
        rows : int
            Satır sayısı
        cols : int
            Sütun sayısı
        figsize : tuple
            Figür boyutu
        vmin : float, optional
            Minimum değer
        vmax : float, optional
            Maksimum değer
        cmap : str
            Renk haritası
        contours : int
            Kontur sayısı
        show_names : bool
            Kanal adlarını göster
        masks : dict, optional
            Panel maskeleme
        titles : dict, optional
            Panel başlıkları
        **kwargs
            Diğer parametreler
            
        Returns
        -------
        plt.Figure
            Çizilen figür
        """
        if self.verbose:
            console.print(f"[blue]Çoklu panel topomap çiziliyor: {rows}x{cols}[/blue]")
            
        # Ortak renk skalası için vmin/vmax hesapla
        if vmin is None or vmax is None:
            all_values = np.concatenate(list(data_dict.values()))
            if vmin is None:
                vmin = np.percentile(all_values, 5)
            if vmax is None:
                vmax = np.percentile(all_values, 95)
                
        # Figür oluştur
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        # Panel verilerini sırala
        panel_names = list(data_dict.keys())
        
        for i, panel_name in enumerate(panel_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = data_dict[panel_name]
            mask = masks.get(panel_name) if masks else None
            title = titles.get(panel_name, panel_name) if titles else panel_name
            
            # Topomap çiz
            im, _ = mne.viz.plot_topomap(
                values,
                info,
                cmap=cmap,
                contours=contours,
                mask=mask,
                axes=ax,
                show=False,
                **kwargs
            )
            
            # Renk skalasını manuel olarak ayarla
            if vmin is not None and vmax is not None:
                im.set_clim(vmin=vmin, vmax=vmax)
            
            # Başlık ekle
            ax.set_title(title, fontsize=10, fontweight='bold')
            
        # Boş panelleri gizle
        for i in range(len(panel_names), len(axes)):
            axes[i].set_visible(False)
            
        # Ortak renk çubuğu ekle
        if im is not None:
            cbar = fig.colorbar(im, ax=axes[:len(panel_names)], shrink=0.8, aspect=20)
            cbar.set_label('Değer', fontsize=10)
            
        plt.tight_layout()
        
        self.fig = fig
        self.axes = axes
        
        return fig
    
    def plot_difference_topomap(
        self,
        group1_values: np.ndarray,
        group2_values: np.ndarray,
        info: mne.Info,
        significance_mask: Optional[np.ndarray] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = 'RdBu_r',
        contours: int = 6,
        show_names: bool = False,
        title: str = 'Grup Farkı',
        **kwargs
    ) -> plt.Figure:
        """
        Grup farkı topolojik haritası çiz
        
        Parameters
        ----------
        group1_values : np.ndarray
            Birinci grup değerleri
        group2_values : np.ndarray
            İkinci grup değerleri
        info : mne.Info
            Kanal bilgileri
        significance_mask : np.ndarray, optional
            Anlamlılık maskesi
        vmin : float, optional
            Minimum değer
        vmax : float, optional
            Maksimum değer
        cmap : str
            Renk haritası
        contours : int
            Kontur sayısı
        show_names : bool
            Kanal adlarını göster
        title : str
            Başlık
        **kwargs
            Diğer parametreler
            
        Returns
        -------
        plt.Figure
            Çizilen figür
        """
        if self.verbose:
            console.print("[blue]Fark topolojik haritası çiziliyor...[/blue]")
            
        # Fark hesapla
        difference = group1_values - group2_values
        
        # Varsayılan vmin/vmax
        if vmin is None:
            vmin = -np.max(np.abs(difference))
        if vmax is None:
            vmax = np.max(np.abs(difference))
            
        # Figür oluştur
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Topomap çiz
        im, _ = mne.viz.plot_topomap(
            difference,
            info,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            contours=contours,
            show_names=show_names,
            mask=significance_mask,
            axes=ax,
            show=False,
            **kwargs
        )
        
        # Başlık ekle
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Renk çubuğu ekle
        if im is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Fark (Grup1 - Grup2)', fontsize=10)
            
        return fig
    
    def plot_statistical_mask(
        self,
        p_values: np.ndarray,
        info: mne.Info,
        alpha: float = 0.05,
        corrected: bool = False,
        title: str = 'İstatistiksel Anlamlılık',
        **kwargs
    ) -> plt.Figure:
        """
        İstatistiksel anlamlılık maskesi çiz
        
        Parameters
        ----------
        p_values : np.ndarray
            P-değerleri
        info : mne.Info
            Kanal bilgileri
        alpha : float
            Anlamlılık düzeyi
        corrected : bool
            Düzeltilmiş p-değerleri
        title : str
            Başlık
        **kwargs
            Diğer parametreler
            
        Returns
        -------
        plt.Figure
            Çizilen figür
        """
        if self.verbose:
            console.print("[blue]İstatistiksel maske çiziliyor...[/blue]")
            
        # Anlamlılık maskesi
        significant = p_values < alpha
        
        # Figür oluştur
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Topomap çiz
        im, _ = mne.viz.plot_topomap(
            p_values,
            info,
            vmin=0,
            vmax=alpha,
            cmap='Reds',
            contours=6,
            show_names=False,
            mask=significant,
            mask_params={
                'marker': 'o',
                'markerfacecolor': 'w',
                'markeredgecolor': 'k',
                'markersize': 4,
                'linewidth': 0
            },
            axes=ax,
            show=False,
            **kwargs
        )
        
        # Başlık ekle
        correction_text = " (Düzeltilmiş)" if corrected else ""
        ax.set_title(f"{title}{correction_text}", fontsize=12, fontweight='bold')
        
        # Renk çubuğu ekle
        if im is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('P-değeri', fontsize=10)
            
        return fig
    
    def plot_metric_comparison(
        self,
        metrics_data: Dict[str, np.ndarray],
        info: mne.Info,
        metric_names: List[str],
        rows: int = 2,
        cols: int = 2,
        figsize: Tuple[float, float] = (12, 8),
        **kwargs
    ) -> plt.Figure:
        """
        Metrik karşılaştırma çizimi
        
        Parameters
        ----------
        metrics_data : dict
            Metrik verileri
        info : mne.Info
            Kanal bilgileri
        metric_names : list
            Metrik adları
        rows : int
            Satır sayısı
        cols : int
            Sütun sayısı
        figsize : tuple
            Figür boyutu
        **kwargs
            Diğer parametreler
            
        Returns
        -------
        plt.Figure
            Çizilen figür
        """
        if self.verbose:
            console.print("[blue]Metrik karşılaştırması çiziliyor...[/blue]")
            
        # Figür oluştur
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for i, metric_name in enumerate(metric_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = metrics_data[metric_name]
            
            # Topomap çiz
            im, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                show=False,
                **kwargs
            )
            
            # Başlık ekle
            ax.set_title(metric_name, fontsize=10, fontweight='bold')
            
        # Boş panelleri gizle
        for i in range(len(metric_names), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        
        return fig
    
    def set_publication_style(
        self,
        figsize: Tuple[float, float] = (6, 6),
        dpi: int = 300,
        font_size: int = 12,
        font_family: str = 'serif'
    ):
        """
        Yayın kalitesi stil ayarları
        
        Parameters
        ----------
        figsize : tuple
            Figür boyutu (mm cinsinden)
        dpi : int
            Çözünürlük
        font_size : int
            Font boyutu
        font_family : str
            Font ailesi
        """
        # Matplotlib stil ayarları
        plt.rcParams.update({
            'figure.figsize': figsize,
            'figure.dpi': dpi,
            'font.size': font_size,
            'font.family': font_family,
            'axes.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'legend.frameon': False,
            'legend.fontsize': font_size - 2,
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        if self.verbose:
            console.print("[green]Yayın kalitesi stil ayarlandı[/green]")
    
    def create_colorbar(
        self,
        vmin: float,
        vmax: float,
        cmap: str = 'viridis',
        label: str = 'Değer',
        orientation: str = 'vertical'
    ) -> plt.Figure:
        """
        Renk çubuğu oluştur
        
        Parameters
        ----------
        vmin : float
            Minimum değer
        vmax : float
            Maksimum değer
        cmap : str
            Renk haritası
        label : str
            Etiket
        orientation : str
            Yönlendirme
            
        Returns
        -------
        plt.Figure
            Renk çubuğu figürü
        """
        fig, ax = plt.subplots(1, 1, figsize=(1, 4))
        
        # Renk haritası
        cmap_obj = cm.get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Renk çubuğu
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=ax, orientation=orientation)
        cbar.set_label(label, fontsize=10)
        
        return fig


def create_topomap_grid(
    data_dict: Dict[str, np.ndarray],
    info: mne.Info,
    grid_shape: Tuple[int, int] = (2, 2),
    titles: Optional[Dict[str, str]] = None,
    **kwargs
) -> plt.Figure:
    """
    Topolojik harita grid'i oluştur
    
    Parameters
    ----------
    data_dict : dict
        Panel verileri
    info : mne.Info
        Kanal bilgileri
    grid_shape : tuple
        Grid boyutu (rows, cols)
    titles : dict, optional
        Panel başlıkları
    **kwargs
        Diğer parametreler
        
    Returns
    -------
    plt.Figure
        Grid figürü
    """
    visualizer = EEGVisualizer()
    
    return visualizer.plot_multi_panel_topomap(
        data_dict,
        info,
        rows=grid_shape[0],
        cols=grid_shape[1],
        titles=titles,
        **kwargs
    )


def plot_metric_distribution(
    values: np.ndarray,
    channel_names: List[str],
    metric_name: str = 'Metrik',
    bins: int = 30
) -> plt.Figure:
    """
    Metrik dağılımı çiz
    
    Parameters
    ----------
    values : np.ndarray
        Metrik değerleri
    channel_names : list
        Kanal adları
    metric_name : str
        Metrik adı
    bins : int
        Histogram bin sayısı
        
    Returns
    -------
    plt.Figure
        Dağılım figürü
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(values, bins=bins, alpha=0.7, edgecolor='black')
    ax1.set_xlabel(metric_name)
    ax1.set_ylabel('Frekans')
    ax1.set_title(f'{metric_name} Dağılımı')
    
    # Box plot
    ax2.boxplot(values, patch_artist=True)
    ax2.set_ylabel(metric_name)
    ax2.set_title(f'{metric_name} Box Plot')
    
    plt.tight_layout()
    
    return fig
