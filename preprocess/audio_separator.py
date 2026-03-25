# preprocess/audio_separator.py

import numpy as np
import librosa
from sklearn.decomposition import NMF
from .config import SAMPLE_RATE

class AudioSeparatorWrapper:
    def __init__(self, n_sources=2):
        """
        Initializes an NMF-based Audio Separator.
        This uses basic mathematical factorization (NMF) to find underlying 'sources'
        in the audio, avoiding massive deep learning model downloads or crashes.
        """
        self.n_sources = n_sources
        print(f"Initializing Math-based NMF Separator (components={self.n_sources})...")

    def separate(self, audio_array):
        """
        Takes a 1D numpy array, separates it into `n_sources` arrays using
        spectrogram decomposition, and returns a list of resulting 1D arrays.
        """
        if len(audio_array.shape) != 1:
            raise ValueError("Expected a 1D mono audio array.")

        # 1. Compute the magnitude spectrogram (STFT)
        # Using a standard STFT for speech/bird calls
        stft_matrix = librosa.stft(audio_array, n_fft=2048, hop_length=512)
        magnitude, phase = librosa.magphase(stft_matrix)

        # 2. If the audio is completely silent or flat, return it unseparated safely
        if np.max(magnitude) == 0:
            return [audio_array]

        # 3. Apply Non-negative Matrix Factorization (NMF)
        # We decompose the magnitude spectrogram into exactly `n_sources`
        nmf = NMF(n_components=self.n_sources, init='random', random_state=42, max_iter=200)
        
        # Fit NMF on the transposed matrix (time x frequencies)
        W = nmf.fit_transform(magnitude.T) # Activations over time
        H = nmf.components_                # Frequency basis for each source

        separated_arrays = []

        # 4. Reconstruct the separated audio waveforms
        for i in range(self.n_sources):
            # Isolate the magnitude for this single source
            # By multiplying its specific time activation (W[:, i]) by its frequency basis (H[i, :])
            source_magnitude = np.outer(H[i, :], W[:, i])

            # Apply a "soft mask" to the original magnitude to minimize artifacts
            mask = source_magnitude / (np.dot(H.T, W.T) + 1e-10)
            masked_magnitude = magnitude * mask

            # Reconstruct the complex spectrogram using the original phase
            source_stft = masked_magnitude * phase

            # Inverse STFT to get the audio waveform back
            source_audio = librosa.istft(source_stft, hop_length=512)
            
            # Ensure the output length strictly matches the input length
            if len(source_audio) > len(audio_array):
                source_audio = source_audio[:len(audio_array)]
            elif len(source_audio) < len(audio_array):
                source_audio = np.pad(source_audio, (0, len(audio_array) - len(source_audio)))

            separated_arrays.append(source_audio)

        return separated_arrays
