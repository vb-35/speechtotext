import os
import soundfile as sf

import librosa
import numpy as np

class AudioLoader:
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate

    def load(self, path):
        # Load audio file and preprocess it
        audio, file_samplerate = sf.read(path)
        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if file_samplerate != self.samplerate:
            audio = librosa.resample(y=audio, orig_sr=file_samplerate, target_sr=self.samplerate)

        return audio
        

    def convert_to_spec(self, audio):
        # Convert audio to spectrogram
        spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.samplerate,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        # Convert to log scale
        spec = librosa.power_to_db(spec, ref=np.max)
        return spec
        pass
    
