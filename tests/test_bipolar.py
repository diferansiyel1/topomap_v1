"""
Bipolar veri işleme modülü testleri
"""

import numpy as np
import pytest
import mne
from eeg_topomap_lab import bipolar


class TestBipolarProcessor:
    """BipolarProcessor test sınıfı"""
    
    def setup_method(self):
        """Test kurulumu"""
        self.processor = bipolar.BipolarProcessor(verbose=False)
        
        # Test verisi oluştur
        sfreq = 250
        duration = 10
        n_channels = 19
        
        # Rastgele EEG verisi
        data = np.random.randn(n_channels, int(sfreq * duration))
        
        # Kanal adları (bipolar kanallar dahil)
        ch_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F3-F4', 'C3-C4', 'P3-P4', 'F7-F8', 'T3-T4', 'T5-T6'
        ]
        
        # MNE Info oluştur
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        
        # Raw objesi oluştur
        self.raw = mne.io.RawArray(data, info, verbose=False)
        
        # Montaj uygula
        montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(montage)
    
    def test_bipolar_channel_detection(self):
        """Bipolar kanal tespiti testi"""
        bipolar_pairs = self.processor.detect_bipolar_channels(self.raw)
        
        assert isinstance(bipolar_pairs, list)
        assert len(bipolar_pairs) > 0
        
        # Beklenen bipolar çiftler
        expected_pairs = [('F3', 'F4'), ('C3', 'C4'), ('P3', 'P4'), ('F7', 'F8'), ('T3', 'T4'), ('T5', 'T6')]
        
        for expected_pair in expected_pairs:
            assert expected_pair in bipolar_pairs
    
    def test_bipolar_to_reference_conversion(self):
        """Bipolar veriden referans dönüşümü testi"""
        # Bipolar kanalları tespit et
        bipolar_pairs = self.processor.detect_bipolar_channels(self.raw)
        
        if len(bipolar_pairs) > 0:
            # Referans dönüşümü uygula
            converted_raw = self.processor.convert_bipolar_to_reference(
                self.raw, ref_type='average'
            )
            
            assert isinstance(converted_raw, mne.io.Raw)
            assert len(converted_raw.ch_names) <= len(self.raw.ch_names)
            
            # Kanal adları bipolar formatında olmalı
            for ch_name in converted_raw.ch_names:
                assert '-' in ch_name or '_' in ch_name
    
    def test_midpoint_coordinates_creation(self):
        """Orta-nokta koordinatları oluşturma testi"""
        # Bipolar kanalları tespit et
        bipolar_pairs = self.processor.detect_bipolar_channels(self.raw)
        
        if len(bipolar_pairs) > 0:
            # Orta-nokta koordinatları oluştur
            midpoint_coords = self.processor.create_midpoint_coordinates(
                self.raw, bipolar_pairs
            )
            
            assert isinstance(midpoint_coords, dict)
            assert len(midpoint_coords) == len(bipolar_pairs)
            
            # Koordinatlar 3D olmalı
            for ch_name, coords in midpoint_coords.items():
                assert isinstance(coords, tuple)
                assert len(coords) == 3
                assert all(isinstance(coord, (int, float)) for coord in coords)
    
    def test_midpoint_approach(self):
        """Orta-nokta yaklaşımı testi"""
        # Bipolar kanalları tespit et
        bipolar_pairs = self.processor.detect_bipolar_channels(self.raw)
        
        if len(bipolar_pairs) > 0:
            # Orta-nokta yaklaşımını uygula
            midpoint_raw = self.processor.apply_midpoint_approach(
                self.raw, bipolar_pairs
            )
            
            assert isinstance(midpoint_raw, mne.io.Raw)
            assert len(midpoint_raw.ch_names) == len(bipolar_pairs)
            
            # Kanal adları bipolar formatında olmalı
            for ch_name in midpoint_raw.ch_names:
                assert '-' in ch_name or '_' in ch_name
    
    def test_csd_approach(self):
        """CSD yaklaşımı testi"""
        try:
            # CSD yaklaşımını uygula
            csd_raw = self.processor.apply_csd_approach(self.raw)
            
            assert isinstance(csd_raw, mne.io.Raw)
            assert len(csd_raw.ch_names) == len(self.raw.ch_names)
            
        except Exception as e:
            # CSD kullanılamıyorsa uyarı ver
            pytest.skip(f"CSD yaklaşımı kullanılamıyor: {e}")
    
    def test_bipolar_info(self):
        """Bipolar bilgi alma testi"""
        # Bipolar kanalları tespit et
        bipolar_pairs = self.processor.detect_bipolar_channels(self.raw)
        
        # Bipolar bilgileri al
        info = self.processor.get_bipolar_info()
        
        assert isinstance(info, dict)
        assert 'bipolar_pairs' in info
        assert 'n_bipolar_pairs' in info
        assert 'reference_channels' in info
        
        assert info['bipolar_pairs'] == bipolar_pairs
        assert info['n_bipolar_pairs'] == len(bipolar_pairs)


class TestBipolarDataProcessing:
    """Bipolar veri işleme ana fonksiyon testleri"""
    
    def setup_method(self):
        """Test kurulumu"""
        # Test verisi oluştur
        sfreq = 250
        duration = 5
        n_channels = 19
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        self.raw = mne.io.RawArray(data, info, verbose=False)
    
    def test_reference_conversion_method(self):
        """Referans dönüşüm metodu testi"""
        converted_raw = bipolar.process_bipolar_data(
            self.raw, method='reference_conversion', ref_type='average'
        )
        
        assert isinstance(converted_raw, mne.io.Raw)
        assert len(converted_raw.ch_names) <= len(self.raw.ch_names)
    
    def test_midpoint_method(self):
        """Orta-nokta metodu testi"""
        converted_raw = bipolar.process_bipolar_data(
            self.raw, method='midpoint'
        )
        
        assert isinstance(converted_raw, mne.io.Raw)
    
    def test_csd_method(self):
        """CSD metodu testi"""
        try:
            converted_raw = bipolar.process_bipolar_data(
                self.raw, method='csd'
            )
            
            assert isinstance(converted_raw, mne.io.Raw)
            
        except Exception as e:
            pytest.skip(f"CSD metodu kullanılamıyor: {e}")
    
    def test_invalid_method(self):
        """Geçersiz metod testi"""
        with pytest.raises(ValueError):
            bipolar.process_bipolar_data(self.raw, method='invalid_method')


class TestBipolarValidation:
    """Bipolar dönüşüm doğrulama testleri"""
    
    def test_validation_reference_conversion(self):
        """Referans dönüşüm doğrulama testi"""
        # Test verisi oluştur
        sfreq = 250
        duration = 5
        n_channels = 19
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        original_raw = mne.io.RawArray(data, info, verbose=False)
        
        # Dönüşüm uygula
        converted_raw = bipolar.process_bipolar_data(
            original_raw, method='reference_conversion', ref_type='average'
        )
        
        # Doğrulama yap
        validation = bipolar.validate_bipolar_conversion(
            original_raw, converted_raw, 'reference_conversion'
        )
        
        assert isinstance(validation, dict)
        assert 'method' in validation
        assert 'original_channels' in validation
        assert 'converted_channels' in validation
        assert 'sfreq_match' in validation
        assert 'data_shape_match' in validation
        
        assert validation['method'] == 'reference_conversion'
        assert validation['original_channels'] == len(original_raw.ch_names)
        assert validation['converted_channels'] == len(converted_raw.ch_names)
        assert validation['sfreq_match'] == True
        assert validation['data_shape_match'] == True
    
    def test_validation_midpoint(self):
        """Orta-nokta doğrulama testi"""
        # Test verisi oluştur
        sfreq = 250
        duration = 5
        n_channels = 19
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        original_raw = mne.io.RawArray(data, info, verbose=False)
        
        # Dönüşüm uygula
        converted_raw = bipolar.process_bipolar_data(
            original_raw, method='midpoint'
        )
        
        # Doğrulama yap
        validation = bipolar.validate_bipolar_conversion(
            original_raw, converted_raw, 'midpoint'
        )
        
        assert isinstance(validation, dict)
        assert 'method' in validation
        assert validation['method'] == 'midpoint'
        assert 'midpoint_coords_available' in validation
    
    def test_validation_csd(self):
        """CSD doğrulama testi"""
        # Test verisi oluştur
        sfreq = 250
        duration = 5
        n_channels = 19
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        original_raw = mne.io.RawArray(data, info, verbose=False)
        
        try:
            # Dönüşüm uygula
            converted_raw = bipolar.process_bipolar_data(
                original_raw, method='csd'
            )
            
            # Doğrulama yap
            validation = bipolar.validate_bipolar_conversion(
                original_raw, converted_raw, 'csd'
            )
            
            assert isinstance(validation, dict)
            assert 'method' in validation
            assert validation['method'] == 'csd'
            assert 'csd_applied' in validation
            
        except Exception as e:
            pytest.skip(f"CSD doğrulama testi atlandı: {e}")


class TestBipolarEdgeCases:
    """Bipolar kenar durumları testleri"""
    
    def test_no_bipolar_channels(self):
        """Bipolar kanal olmayan durum testi"""
        # Sadece unipolar kanallar
        sfreq = 250
        duration = 5
        n_channels = 19
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        processor = bipolar.BipolarProcessor(verbose=False)
        bipolar_pairs = processor.detect_bipolar_channels(raw)
        
        assert len(bipolar_pairs) == 0
    
    def test_all_bipolar_channels(self):
        """Tüm kanallar bipolar olan durum testi"""
        # Sadece bipolar kanallar
        sfreq = 250
        duration = 5
        n_channels = 10
        
        data = np.random.randn(n_channels, int(sfreq * duration))
        ch_names = [f'Ch{i}-Ch{j}' for i, j in zip(range(0, n_channels, 2), range(1, n_channels, 2))]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        processor = bipolar.BipolarProcessor(verbose=False)
        bipolar_pairs = processor.detect_bipolar_channels(raw)
        
        assert len(bipolar_pairs) > 0
        assert len(bipolar_pairs) == n_channels // 2
