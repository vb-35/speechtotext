import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioLoader:
    """Audio loading and processing class with support for multiple formats and resampling."""
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac'}
    DEFAULT_SAMPLE_RATE = 16000  # 16kHz is common for speech recognition
    
    def __init__(self, target_sample_rate: int = DEFAULT_SAMPLE_RATE):
        """
        Initialize the AudioLoader with target sample rate.
        
        Args:
            target_sample_rate (int): Target sample rate for all audio files
        """
        self.target_sample_rate = target_sample_rate
        
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file and convert to mono if necessary.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple containing:
                - Audio tensor (shape: [1, num_samples])
                - Sample rate
                
        Raises:
            ValueError: If file format is not supported or file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValueError(f"Audio file does not exist: {file_path}")
            
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}. "
                           f"Supported formats are: {self.SUPPORTED_FORMATS}")
        
        try:
            # Load the audio file
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info(f"Converted {file_path.name} to mono")
            
            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                waveform = self._resample_audio(waveform, sample_rate)
                sample_rate = self.target_sample_rate
                
            return waveform, sample_rate
            
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {str(e)}")
    
    def _resample_audio(self, waveform: torch.Tensor, original_sample_rate: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.
        
        Args:
            waveform: Input audio tensor
            original_sample_rate: Original sample rate of the audio
            
        Returns:
            Resampled audio tensor
        """
        resampler = T.Resample(
            orig_freq=original_sample_rate,
            new_freq=self.target_sample_rate
        )
        
        return resampler(waveform)
    
    def get_duration(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """
        Calculate the duration of the audio in seconds.
        
        Args:
            waveform: Audio tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            Duration in seconds
        """
        return waveform.shape[1] / sample_rate
    
    def check_audio_length(self, 
                          waveform: torch.Tensor,
                          sample_rate: int,
                          min_duration: Optional[float] = None,
                          max_duration: Optional[float] = None) -> bool:
        """
        Check if audio duration is within specified limits.
        
        Args:
            waveform: Audio tensor
            sample_rate: Sample rate of the audio
            min_duration: Minimum duration in seconds (optional)
            max_duration: Maximum duration in seconds (optional)
            
        Returns:
            True if duration is within limits, False otherwise
        """
        duration = self.get_duration(waveform, sample_rate)
        
        if min_duration is not None and duration < min_duration:
            logger.warning(f"Audio duration ({duration:.2f}s) is below minimum ({min_duration}s)")
            return False
            
        if max_duration is not None and duration > max_duration:
            logger.warning(f"Audio duration ({duration:.2f}s) exceeds maximum ({max_duration}s)")
            return False
            
        return True