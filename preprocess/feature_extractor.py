# preprocess/feature_extractor.py

import librosa
import numpy as np
from .config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH


class FeatureExtractor:

    def waveform_to_mel(self, audio):
        """
        Step 6:
        Convert waveform → Log-Mel spectrogram
        """

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Optional normalization (stabilizes CNN training)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        return mel_db