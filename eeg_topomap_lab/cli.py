"""
Komut satırı arayüzü

Typer tabanlı CLI arayüzü.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import io, preproc, metrics, bipolar, stats, viz, export

app = typer.Typer(
    name="eegtopo",
    help="EEG topolojik harita analizi için kapsamlı Python paketi",
    add_completion=False
)

console = Console()


@app.command()
def analyze(
    input: str = typer.Option(..., "--input", "-i", help="EEG dosya yolu"),
    montage: str = typer.Option("standard_1020", "--montage", "-m", help="Montaj türü"),
    segments: List[str] = typer.Option([], "--segments", "-s", help="Segment tanımları (örn: 'preiktal:300-600')"),
    metric: str = typer.Option("dfa", "--metric", help="Metrik türü"),
    band: Optional[str] = typer.Option(None, "--band", help="Frekans bandı"),
    compare: Optional[List[str]] = typer.Option(None, "--compare", help="Karşılaştırılacak gruplar"),
    export: str = typer.Option("output.svg", "--export", "-e", help="Çıktı dosya yolu"),
    meta: Optional[str] = typer.Option(None, "--meta", help="Metadata dosya yolu"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Konfigürasyon dosyası"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Detaylı çıktı")
):
    """EEG topolojik harita analizi yap"""
    
    if verbose:
        console.print("[blue]EEG Topomap Lab - Analiz başlatılıyor...[/blue]")
    
    try:
        # Konfigürasyon yükle
        if config:
            with open(config, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            config_data = {}
            
        # Veri yükle
        loader = io.EEGDataLoader(verbose=verbose)
        raw = loader.load_data(input, montage=montage)
        
        # Segmentleri parse et
        segment_dict = {}
        for seg_str in segments:
            if ':' in seg_str:
                name, times = seg_str.split(':', 1)
                if '-' in times:
                    start, end = map(float, times.split('-'))
                    segment_dict[name] = [start, end]
                else:
                    console.print(f"[yellow]Geçersiz segment formatı: {seg_str}[/yellow]")
        
        # Ön işleme
        preprocessor = preproc.EEGPreprocessor(verbose=verbose)
        
        # Filtre uygula
        if 'preprocessing' in config_data:
            preproc_params = config_data['preprocessing']
            raw = preprocessor.apply_filters(
                raw,
                l_freq=preproc_params.get('l_freq'),
                h_freq=preproc_params.get('h_freq'),
                notch_freqs=preproc_params.get('notch_freqs')
            )
            
        # Re-referencing
        ref_type = config_data.get('preprocessing', {}).get('ref_type', 'average')
        raw = preprocessor.apply_reference(raw, ref_type=ref_type)
        
        # Bipolar işleme
        if 'bipolar_processing' in config_data:
            bipolar_params = config_data['bipolar_processing']
            method = bipolar_params.get('method', 'reference_conversion')
            raw = bipolar.process_bipolar_data(raw, method=method)
        
        # Segmentleme
        if segment_dict:
            segmented_data = preprocessor.segment_data(raw, segment_dict)
        else:
            segmented_data = {'all': raw}
        
        # Metrik hesaplama
        metrics_calculator = metrics.EEGMetrics(verbose=verbose)
        
        all_metrics = {}
        for seg_name, seg_raw in segmented_data.items():
            if metric == 'dfa':
                seg_metrics = metrics_calculator.calculate_nonlinear_metrics(
                    seg_raw, metrics=['dfa']
                )
            elif metric == 'band_power':
                seg_metrics = metrics_calculator.calculate_frequency_domain_metrics(
                    seg_raw, metrics=['band_power'], bands={band: (8, 13) if band == 'alpha' else (0.5, 4)}
                )
            else:
                seg_metrics = metrics_calculator.calculate_time_domain_metrics(
                    seg_raw, metrics=[metric]
                )
            
            all_metrics[seg_name] = seg_metrics
        
        # İstatistiksel analiz
        if compare and len(compare) == 2:
            stats_analyzer = stats.EEGStatistics(verbose=verbose)
            
            group1_data = all_metrics[compare[0]][metric]
            group2_data = all_metrics[compare[1]][metric]
            
            test_results = stats_analyzer.two_group_test(
                group1_data, group2_data, test_type='ttest', paired=False
            )
            
            # FDR düzeltmesi
            correction_results = stats_analyzer.multiple_comparison_correction(
                test_results['p_values'], method='fdr', alpha=0.05
            )
            test_results.update(correction_results)
        
        # Görselleştirme
        visualizer = viz.EEGVisualizer(verbose=verbose)
        
        if compare and len(compare) == 2:
            # Fark haritası
            fig = visualizer.plot_difference_topomap(
                all_metrics[compare[0]][metric],
                all_metrics[compare[1]][metric],
                raw.info,
                significance_mask=test_results.get('significant'),
                title=f"{compare[0]} vs {compare[1]} - {metric}"
            )
        else:
            # Tek panel topomap
            if len(segmented_data) == 1:
                seg_name = list(segmented_data.keys())[0]
                values = all_metrics[seg_name][metric]
                fig = visualizer.plot_topomap(
                    values, raw.info, title=f"{seg_name} - {metric}"
                )
            else:
                # Çoklu panel
                panel_data = {name: metrics[metric] for name, metrics in all_metrics.items()}
                fig = visualizer.plot_multi_panel_topomap(
                    panel_data, raw.info, rows=2, cols=2
                )
        
        # Dışa aktarma
        exporter = export.EEGExporter(verbose=verbose)
        
        # Figürü kaydet
        success = exporter.export_figure(fig, export)
        
        if success:
            console.print(f"[green]Analiz tamamlandı: {export}[/green]")
        else:
            console.print("[red]Analiz başarısız[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Hata: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    input: str = typer.Option(..., "--input", "-i", help="EEG dosya yolu"),
    output: str = typer.Option("config.yaml", "--output", "-o", help="Konfigürasyon dosya yolu"),
    template: str = typer.Option("basic", "--template", "-t", help="Konfigürasyon şablonu")
):
    """Konfigürasyon dosyası oluştur"""
    
    # Şablon konfigürasyonlar
    templates = {
        "basic": {
            "input": {
                "file": input,
                "montage": "standard_1020"
            },
            "segments": {
                "preiktal": [300, 600],
                "interiktal": [1200, 1500]
            },
            "metrics": {
                "type": "dfa",
                "dfa_min": 10,
                "dfa_max": 100
            },
            "statistics": {
                "compare": ["preiktal", "interiktal"],
                "paired": False,
                "fdr": 0.05
            },
            "visualization": {
                "vmin": 0.5,
                "vmax": 1.2,
                "cmap": "viridis",
                "contours": 0,
                "show_names": False
            },
            "export": {
                "figure": "fig/out_pre_vs_inter_alpha_dfa.svg",
                "metadata": "fig/out_pre_vs_inter_alpha_dfa.json"
            }
        },
        "frequency": {
            "input": {
                "file": input,
                "montage": "standard_1020"
            },
            "segments": {
                "rest": [0, 300],
                "task": [300, 600]
            },
            "metrics": {
                "type": "band_power",
                "band": "alpha",
                "relative": True
            },
            "statistics": {
                "compare": ["rest", "task"],
                "paired": True,
                "fdr": 0.05
            },
            "visualization": {
                "vmin": 0.0,
                "vmax": 0.6,
                "cmap": "viridis",
                "contours": 6,
                "show_names": True
            },
            "export": {
                "figure": "fig/alpha_rel_rest_vs_task.svg",
                "metadata": "fig/alpha_rel_rest_vs_task.json"
            }
        }
    }
    
    if template not in templates:
        console.print(f"[red]Bilinmeyen şablon: {template}[/red]")
        console.print(f"Kullanılabilir şablonlar: {', '.join(templates.keys())}")
        raise typer.Exit(1)
    
    # Konfigürasyon dosyasını oluştur
    config_data = templates[template]
    config_data["input"]["file"] = input
    
    # YAML olarak kaydet
    import yaml
    with open(output, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    console.print(f"[green]Konfigürasyon oluşturuldu: {output}[/green]")


@app.command()
def info(
    input: str = typer.Option(..., "--input", "-i", help="EEG dosya yolu"),
    montage: str = typer.Option("standard_1020", "--montage", "-m", help="Montaj türü")
):
    """EEG dosya bilgilerini göster"""
    
    try:
        # Veri yükle
        loader = io.EEGDataLoader(verbose=False)
        raw = loader.load_data(input, montage=montage)
        
        # Bilgi tablosu oluştur
        table = Table(title="EEG Dosya Bilgileri")
        table.add_column("Özellik", style="cyan")
        table.add_column("Değer", style="magenta")
        
        table.add_row("Dosya", str(input))
        table.add_row("Kanal sayısı", str(len(raw.ch_names)))
        table.add_row("Örnekleme frekansı", f"{raw.info['sfreq']} Hz")
        table.add_row("Süre", f"{raw.times[-1]:.1f} saniye")
        table.add_row("Montaj", montage)
        
        # Kanal bilgileri
        console.print(table)
        
        # Kanal listesi
        if len(raw.ch_names) <= 20:
            console.print("\n[blue]Kanal listesi:[/blue]")
            console.print(", ".join(raw.ch_names))
        else:
            console.print(f"\n[blue]Kanal sayısı: {len(raw.ch_names)}[/blue]")
            console.print("İlk 10 kanal:", ", ".join(raw.ch_names[:10]))
            console.print("Son 10 kanal:", ", ".join(raw.ch_names[-10:]))
        
        # Bipolar kanal tespiti
        processor = bipolar.BipolarProcessor(verbose=False)
        bipolar_pairs = processor.detect_bipolar_channels(raw)
        
        if bipolar_pairs:
            console.print(f"\n[blue]Bipolar kanal çiftleri ({len(bipolar_pairs)}):[/blue]")
            for ch1, ch2 in bipolar_pairs[:10]:  # İlk 10 çift
                console.print(f"  {ch1} - {ch2}")
            if len(bipolar_pairs) > 10:
                console.print(f"  ... ve {len(bipolar_pairs) - 10} tane daha")
        
    except Exception as e:
        console.print(f"[red]Hata: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    input: str = typer.Option(..., "--input", "-i", help="EEG dosya yolu"),
    montage: str = typer.Option("standard_1020", "--montage", "-m", help="Montaj türü")
):
    """EEG dosyasını doğrula"""
    
    try:
        console.print("[blue]EEG dosyası doğrulanıyor...[/blue]")
        
        # Veri yükle
        loader = io.EEGDataLoader(verbose=False)
        raw = loader.load_data(input, montage=montage)
        
        # Temel doğrulamalar
        issues = []
        warnings = []
        
        # Kanal sayısı kontrolü
        if len(raw.ch_names) < 8:
            issues.append("Çok az kanal (< 8)")
        elif len(raw.ch_names) > 256:
            warnings.append("Çok fazla kanal (> 256)")
        
        # Örnekleme frekansı kontrolü
        if raw.info['sfreq'] < 100:
            warnings.append("Düşük örnekleme frekansı (< 100 Hz)")
        elif raw.info['sfreq'] > 1000:
            warnings.append("Yüksek örnekleme frekansı (> 1000 Hz)")
        
        # Süre kontrolü
        duration = raw.times[-1]
        if duration < 60:
            warnings.append("Kısa kayıt süresi (< 60 saniye)")
        elif duration > 3600:
            warnings.append("Uzun kayıt süresi (> 1 saat)")
        
        # Montaj kontrolü
        if not raw.get_montage():
            issues.append("Montaj bilgisi bulunamadı")
        
        # Sonuçları göster
        if issues:
            console.print("[red]Kritik sorunlar:[/red]")
            for issue in issues:
                console.print(f"  ❌ {issue}")
        
        if warnings:
            console.print("[yellow]Uyarılar:[/yellow]")
            for warning in warnings:
                console.print(f"  ⚠️  {warning}")
        
        if not issues and not warnings:
            console.print("[green]Dosya doğrulandı - sorun bulunamadı[/green]")
        elif not issues:
            console.print("[green]Dosya doğrulandı - sadece uyarılar var[/green]")
        else:
            console.print("[red]Dosya doğrulanamadı - kritik sorunlar var[/red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]Doğrulama hatası: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_templates():
    """Kullanılabilir şablonları listele"""
    
    templates = {
        "basic": "Temel DFA analizi (preiktal vs interiktal)",
        "frequency": "Frekans domeni analizi (alpha band gücü)",
        "time_domain": "Zaman domeni metrikleri (RMS, tepe-tepe)",
        "nonlinear": "Nonlineer metrikler (DFA, entropy)",
        "multi_group": "Çoklu grup karşılaştırması"
    }
    
    table = Table(title="Kullanılabilir Şablonlar")
    table.add_column("Şablon", style="cyan")
    table.add_column("Açıklama", style="magenta")
    
    for template, description in templates.items():
        table.add_row(template, description)
    
    console.print(table)


def main():
    """Ana CLI fonksiyonu"""
    app()


if __name__ == "__main__":
    main()
