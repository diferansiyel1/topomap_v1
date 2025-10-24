"""
Pytest konfigürasyonu ve fixture'lar
"""

import numpy as np
import pytest
import mne
from pathlib import Path
import tempfile


@pytest.fixture
def sample_eeg_data():
    """Örnek EEG verisi fixture'ı"""
    # Test parametreleri
    sfreq = 250
    duration = 10
    n_channels = 19
    
    # Rastgele EEG verisi oluştur
    np.random.seed(42)
    data = np.random.randn(n_channels, int(sfreq * duration))
    
    # Kanal adları (10-20 sistemi)
    ch_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    
    # MNE Info oluştur
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    
    # Raw objesi oluştur
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Montaj uygula
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    return raw


@pytest.fixture
def bipolar_eeg_data():
    """Bipolar EEG verisi fixture'ı"""
    # Test parametreleri
    sfreq = 250
    duration = 10
    n_channels = 16
    
    # Rastgele EEG verisi oluştur
    np.random.seed(42)
    data = np.random.randn(n_channels, int(sfreq * duration))
    
    # Bipolar kanal adları
    ch_names = [
        'F3-F4', 'C3-C4', 'P3-P4', 'O1-O2',
        'F7-F8', 'T3-T4', 'T5-T6', 'Fz-Cz',
        'Cz-Pz', 'Fp1-Fp2', 'F3-C3', 'C3-P3',
        'P3-O1', 'F4-C4', 'C4-P4', 'P4-O2'
    ]
    
    # MNE Info oluştur
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    
    # Raw objesi oluştur
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Montaj uygula
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    return raw


@pytest.fixture
def sample_metrics_data():
    """Örnek metrik verisi fixture'ı"""
    n_channels = 19
    
    metrics_data = {
        'rms': np.random.randn(n_channels),
        'mean': np.random.randn(n_channels),
        'peak_to_peak': np.random.randn(n_channels),
        'dfa': np.random.randn(n_channels),
        'alpha_power': np.random.randn(n_channels),
        'beta_power': np.random.randn(n_channels)
    }
    
    return metrics_data


@pytest.fixture
def sample_statistical_results():
    """Örnek istatistiksel sonuçlar fixture'ı"""
    n_channels = 19
    
    test_results = {
        'test_type': 'independent_ttest',
        'paired': False,
        'statistics': np.random.randn(n_channels),
        'p_values': np.random.uniform(0, 1, n_channels),
        'effect_sizes': np.random.randn(n_channels),
        'corrected_p_values': np.random.uniform(0, 1, n_channels),
        'significant': np.random.choice([True, False], n_channels),
        'n_significant': np.random.randint(0, n_channels)
    }
    
    return test_results


@pytest.fixture
def sample_segments():
    """Örnek segment tanımları fixture'ı"""
    segments = {
        'preiktal': [300, 600],
        'interiktal': [1200, 1500],
        'ictal': [600, 900],
        'postictal': [900, 1200]
    }
    
    return segments


@pytest.fixture
def temp_directory():
    """Geçici dizin fixture'ı"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config():
    """Örnek konfigürasyon fixture'ı"""
    config = {
        'input': {
            'file': 'test.edf',
            'montage': 'standard_1020'
        },
        'segments': {
            'preiktal': [300, 600],
            'interiktal': [1200, 1500]
        },
        'metrics': {
            'type': 'dfa',
            'dfa_min': 10,
            'dfa_max': 100
        },
        'statistics': {
            'compare': ['preiktal', 'interiktal'],
            'paired': False,
            'fdr': 0.05
        },
        'visualization': {
            'vmin': 0.5,
            'vmax': 1.2,
            'cmap': 'viridis',
            'contours': 0,
            'show_names': False
        },
        'export': {
            'figure': 'fig/out_pre_vs_inter_alpha_dfa.svg',
            'metadata': 'fig/out_pre_vs_inter_alpha_dfa.json'
        }
    }
    
    return config


@pytest.fixture
def sample_metadata():
    """Örnek metadata fixture'ı"""
    metadata = {
        'analysis_info': {
            'timestamp': '2024-01-01T00:00:00',
            'software_version': 'eeg-topomap-lab-0.1.0',
            'input_file': 'test.edf',
            'montage': 'standard_1020'
        },
        'segments': {
            'preiktal': [300, 600],
            'interiktal': [1200, 1500]
        },
        'metrics': ['dfa', 'alpha_power'],
        'statistics': {
            'test_type': 'independent_ttest',
            'n_significant': 5,
            'correction_method': 'fdr'
        },
        'visualization': {
            'vmin': 0.5,
            'vmax': 1.2,
            'cmap': 'viridis'
        },
        'preprocessing': {
            'l_freq': 1.0,
            'h_freq': 40.0,
            'ref_type': 'average'
        },
        'bipolar_processing': {
            'method': 'reference_conversion',
            'ref_type': 'average'
        }
    }
    
    return metadata


@pytest.fixture(scope="session")
def test_data_directory():
    """Test veri dizini fixture'ı"""
    # Test veri dizini oluştur
    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)
    
    return test_dir


@pytest.fixture
def mock_eeg_file(temp_directory):
    """Mock EEG dosyası fixture'ı"""
    # Test EEG dosyası oluştur
    sfreq = 250
    duration = 5
    n_channels = 19
    
    # Rastgele EEG verisi
    np.random.seed(42)
    data = np.random.randn(n_channels, int(sfreq * duration))
    
    # Kanal adları
    ch_names = [f'Ch{i}' for i in range(n_channels)]
    
    # MNE Info oluştur
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    
    # Raw objesi oluştur
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Geçici dosya olarak kaydet
    file_path = temp_directory / "test_eeg.fif"
    raw.save(file_path, overwrite=True)
    
    return file_path


# Pytest konfigürasyonu
def pytest_configure(config):
    """Pytest konfigürasyonu"""
    # Test marker'ları tanımla
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Test koleksiyonu değişiklikleri"""
    # Yavaş testleri işaretle
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.integration)
