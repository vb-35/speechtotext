import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib
import matplotlib.pyplot as plt
# Use 'Agg' backend if no display is available
matplotlib.use('Agg')
from typing import Optional, Tuple, Dict
import numpy as np

class FeatureExtractor:
    """Audio feature extraction class supporting spectrograms, MEL features, and visualizations."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,  # 25ms at 16kHz
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: int = 80,
        window_fn: str = "hann",
        power: float = 2.0,
    ):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size (in samples)
            win_length: Window size (in samples). Defaults to n_fft
            hop_length: Hop size between windows. Defaults to win_length // 4
            n_mels: Number of MEL filterbanks
            window_fn: Window function ("hann" or "hamming")
            power: Power of the magnitude spectrogram
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.win_length // 4
        self.n_mels = n_mels
        self.power = power
        
        # Initialize transforms
        window_dict = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window
        }
        self.window_fn = window_dict[window_fn]
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
            window_fn=self.window_fn
        )
        
    def compute_spectrogram(
        self,
        waveform: torch.Tensor,
        return_db: bool = True,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float = 80.0
    ) -> torch.Tensor:
        """
        Compute spectrogram from waveform.
        
        Args:
            waveform: Input audio tensor [1, num_samples]
            return_db: Whether to convert to decibels
            ref: Reference value for db conversion
            amin: Minimum value for db conversion
            top_db: Maximum db value
            
        Returns:
            Spectrogram tensor [1, freq_bins, time_steps]
        """
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_fn(self.win_length),
            return_complex=True
        )
        
        # Compute magnitude spectrogram
        spec = torch.abs(stft).pow(self.power)
        
        if return_db:
            spec = self._to_db(spec, ref=ref, amin=amin, top_db=top_db)
            
        return spec
        
    def compute_melspectrogram(
        self,
        waveform: torch.Tensor,
        return_db: bool = True,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float = 80.0
    ) -> torch.Tensor:
        """
        Compute MEL spectrogram from waveform.
        
        Args:
            waveform: Input audio tensor [1, num_samples]
            return_db: Whether to convert to decibels
            ref: Reference value for db conversion
            amin: Minimum value for db conversion
            top_db: Maximum db value
            
        Returns:
            MEL spectrogram tensor [1, n_mels, time_steps]
        """
        mel_spec = self.mel_transform(waveform)
        
        if return_db:
            mel_spec = self._to_db(mel_spec, ref=ref, amin=amin, top_db=top_db)
            
        return mel_spec
        
    def apply_time_masking(
        self,
        spec: torch.Tensor,
        time_mask_param: int = 30
    ) -> torch.Tensor:
        """Apply time masking augmentation."""
        time_mask = T.TimeMasking(time_mask_param)
        return time_mask(spec)
        
    def apply_freq_masking(
        self,
        spec: torch.Tensor,
        freq_mask_param: int = 20
    ) -> torch.Tensor:
        """Apply frequency masking augmentation."""
        freq_mask = T.FrequencyMasking(freq_mask_param)
        return freq_mask(spec)
        
    def _to_db(
        self,
        spec: torch.Tensor,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float = 80.0
    ) -> torch.Tensor:
        """Convert power/magnitude spectrogram to decibel scale."""
        ref_value = torch.tensor(ref, device=spec.device, dtype=spec.dtype)
        spec = torch.maximum(spec, torch.tensor(amin, device=spec.device, dtype=spec.dtype))
        spec_db = 10.0 * torch.log10(spec / ref_value)
        
        if top_db is not None:
            spec_db = torch.maximum(spec_db, spec_db.max() - top_db)
            
        return spec_db
        
    def plot_waveform(
        self,
        waveform: torch.Tensor,
        title: str = "Waveform",
        figsize: Tuple[int, int] = (10, 4),
        save_path: Optional[str] = None
    ) -> None:
        """Plot waveform visualization."""
        plt.figure(figsize=figsize)
        plt.plot(waveform[0].numpy())
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot: {e}")
            finally:
                plt.close()
        
    def plot_spectrogram(
        self,
        spec: torch.Tensor,
        title: str = "Spectrogram",
        figsize: Tuple[int, int] = (10, 4),
        save_path: Optional[str] = None
    ) -> None:
        """Plot spectrogram visualization."""
        plt.figure(figsize=figsize)
        plt.imshow(spec[0].numpy(), aspect='auto', origin='lower')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Frequency bin")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot: {e}")
            finally:
                plt.close()
        
    def plot_feature_stats(
        self,
        features: torch.Tensor,
        title: str = "Feature Statistics",
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Plot feature statistics and return basic stats.
        
        Args:
            features: Input feature tensor
            title: Plot title
            
        Returns:
            Dictionary containing mean, std, min, max values
        """
        stats = {
            "mean": features.mean().item(),
            "std": features.std().item(),
            "min": features.min().item(),
            "max": features.max().item()
        }
        
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.hist(features.numpy().flatten(), bins=50)
        plt.title(f"{title} - Distribution")
        plt.xlabel("Value")
        plt.ylabel("Count")
        
        plt.subplot(2, 1, 2)
        plt.boxplot(features.numpy().flatten())
        plt.title(f"{title} - Box Plot")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot: {e}")
            finally:
                plt.close()
        
        return stats