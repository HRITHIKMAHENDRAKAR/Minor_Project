# preprocess/audio_processor.py

import librosa
import numpy as np
from .config import SAMPLE_RATE, SAMPLES_PER_FILE


class AudioProcessor:

    def load_audio(self, path):
        """
        Step 1 & 2:
        - Resample to 22050 Hz
        - Convert to mono
        """
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio

    def normalize(self, audio):
        """
        Step 3:
        Amplitude normalization to [-1, 1]
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def trim_silence(self, audio):
        """
        Step 4:
        Remove leading & trailing silence
        """
        audio, _ = librosa.effects.trim(audio, top_db=20)
        return audio

    def fix_length(self, audio):
        """
        Step 5:
        Enforce fixed 5-second duration
        """
        if len(audio) > SAMPLES_PER_FILE:
            audio = audio[:SAMPLES_PER_FILE]
        else:
            padding = SAMPLES_PER_FILE - len(audio)
            audio = np.pad(audio, (0, padding))
        return audio

    def process(self, path):
        """
        Full waveform preprocessing pipeline
        """
        audio = self.load_audio(path)
        audio = self.normalize(audio)
        audio = self.trim_silence(audio)
        audio = self.fix_length(audio)
        return audio