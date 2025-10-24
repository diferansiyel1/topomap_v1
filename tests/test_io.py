"""
Veri giriş/çıkış modülü testleri
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from eeg_topomap_lab import io


class TestEEGDataLoader:
    """EEGDataLoader test sınıfı"""
    
    def test_channel_mapping_creation(self):
        """Kanal eşleme sözlüğü oluşturma testi"""
        current_names = ['Fp1', 'Fp2', 'C3', 'C4']
        target_names = ['FP1', 'FP2', 'C3', 'C4']
        
        mapping = io.create_channel_mapping(current_names, target_names, fuzzy_match=True)
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        assert 'Fp1' in mapping
        assert mapping['Fp1'] == 'FP1'
    
    def test_similarity_calculation(self):
        """Benzerlik hesaplama testi"""
        s1 = "Fp1"
        s2 = "FP1"
        
        similarity = io._calculate_similarity(s1, s2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Fp1 ve FP1 benzer olmalı
    
    def test_channel_mapping_save_load(self):
        """Kanal eşleme kaydetme/yükleme testi"""
        mapping = {'Fp1': 'FP1', 'Fp2': 'FP2', 'C3': 'C3', 'C4': 'C4'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Kaydet
            io.save_channel_mapping(mapping, tmp_path)
            
            # Yükle
            loaded_mapping = io.load_channel_mapping(tmp_path)
            
            assert loaded_mapping == mapping
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestChannelMapping:
    """Kanal eşleme testleri"""
    
    def test_exact_match(self):
        """Tam eşleşme testi"""
        current_names = ['Fp1', 'Fp2', 'C3', 'C4']
        target_names = ['Fp1', 'Fp2', 'C3', 'C4']
        
        mapping = io.create_channel_mapping(current_names, target_names, fuzzy_match=False)
        
        assert len(mapping) == 4
        for name in current_names:
            assert mapping[name] == name
    
    def test_fuzzy_match(self):
        """Yakın eşleşme testi"""
        current_names = ['Fp1', 'Fp2', 'C3', 'C4']
        target_names = ['FP1', 'FP2', 'C3', 'C4']
        
        mapping = io.create_channel_mapping(current_names, target_names, fuzzy_match=True)
        
        assert len(mapping) >= 2  # En az 2 eşleşme olmalı
        assert mapping['Fp1'] == 'FP1'
        assert mapping['Fp2'] == 'FP2'


class TestDataValidation:
    """Veri doğrulama testleri"""
    
    def test_file_extension_detection(self):
        """Dosya uzantısı tespiti testi"""
        test_cases = [
            ('test.edf', '.edf'),
            ('test.bdf', '.bdf'),
            ('test.fif', '.fif'),
            ('test.fif.gz', '.fif'),
            ('test.csv', '.csv')
        ]
        
        for filename, expected_ext in test_cases:
            path = Path(filename)
            ext = path.suffix.lower()
            if ext == '.gz':
                ext = path.with_suffix('').suffix.lower()
            assert ext == expected_ext
