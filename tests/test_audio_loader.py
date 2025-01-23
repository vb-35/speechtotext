import pytest
import torch
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path
from src.audio.loader import AudioLoader

@pytest.fixture
def audio_loader():
    return AudioLoader(target_sample_rate=16000)

@pytest.fixture
def temp_wav_file():
    # Create a temporary WAV file for testing
    duration = 2.0  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a 440 Hz sine wave
    audio_data = np.sin(2 * np.pi * 440 * t)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio_data, sample_rate)
        return Path(f.name)

def test_load_audio_basic(audio_loader, temp_wav_file):
    waveform, sample_rate = audio_loader.load_audio(temp_wav_file)
    
    assert isinstance(waveform, torch.Tensor)
    assert waveform.dim() == 2  # [channels, samples]
    assert waveform.shape[0] == 1  # mono
    assert sample_rate == audio_loader.target_sample_rate

def test_load_audio_invalid_path(audio_loader):
    with pytest.raises(ValueError, match="Audio file does not exist"):
        audio_loader.load_audio("nonexistent_file.wav")

def test_load_audio_invalid_format(audio_loader):
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        with pytest.raises(ValueError, match="Unsupported audio format"):
            audio_loader.load_audio(f.name)

def test_audio_duration(audio_loader, temp_wav_file):
    waveform, sample_rate = audio_loader.load_audio(temp_wav_file)
    duration = audio_loader.get_duration(waveform, sample_rate)
    
    assert isinstance(duration, float)
    assert abs(duration - 2.0) < 0.1  # approximately 2 seconds

def test_check_audio_length(audio_loader, temp_wav_file):
    waveform, sample_rate = audio_loader.load_audio(temp_wav_file)
    
    # Test within limits
    assert audio_loader.check_audio_length(waveform, sample_rate, min_duration=1.0, max_duration=3.0)
    
    # Test below minimum
    assert not audio_loader.check_audio_length(waveform, sample_rate, min_duration=5.0)
    
    # Test above maximum
    assert not audio_loader.check_audio_length(waveform, sample_rate, max_duration=1.0)