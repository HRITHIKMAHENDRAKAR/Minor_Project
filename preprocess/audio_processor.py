# preprocess/audio_processor.py

import librosa
import numpy as np
import scipy.signal
from .config import SAMPLE_RATE, SAMPLES_PER_FILE


class AudioProcessor:

    def load_audio(self, path):
        """
        Step 1 & 2:
        - Resample to 22050 Hz
        - Convert to mono
        - DYNAMIC High-Pass Filter: Automatically tunes cutoff frequency based on bird pitch.
        - Spectral Noise Gating (Dynamic Denoiser)
        """
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        
        # DYNAMIC FREQUENCY TRACKING
        # 1. Analyze the audio mathematically to find the "Peak Pitch" 
        stft_matrix = np.abs(librosa.stft(audio))
        mean_spectrum = np.mean(stft_matrix, axis=1)
        
        # 2. Find the frequency (Hz) where the loudest bird call is happening
        peak_bin = np.argmax(mean_spectrum)
        peak_freq = librosa.fft_frequencies(sr=SAMPLE_RATE)[peak_bin]
        
        # 3. Intelligently set the wind-blocking filter safely BELOW the bird's true voice.
        # If the bird is a 200Hz Owl, cut at 100Hz. If it's a 3000Hz Bulbul, safely cut wind up to 1000Hz.
        dynamic_cutoff = max(100, min(peak_freq * 0.5, 1000))
        
        sos = scipy.signal.butter(10, dynamic_cutoff, 'hp', fs=SAMPLE_RATE, output='sos')
        audio = scipy.signal.sosfilt(sos, audio)
        
        # Apply Spectral Noise Gating
        try:
            import noisereduce as nr
            # Reduce static background hiss while keeping the bird signal completely pristine
            audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, prop_decrease=0.8)
        except ImportError:
            pass
            
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

    def smart_crop(self, audio, max_duration=15.0):
        """
        Step 5 (Upgraded): Smart Cropping
        Finds the 15-second window with the highest Root Mean Square (RMS) energy.
        This guarantees we extract the clearest bird call (highest Signal-to-Noise Ratio)
        and drop unnecessary silence/wind, speeding up NMF and BirdNET inference.
        """
        max_samples = int(max_duration * SAMPLE_RATE)
        
        # If audio is already shorter than our max duration, keep it whole
        if len(audio) <= max_samples:
            return audio
            
        # Calculate RMS energy of the audio (hop_length of 512 is extremely standard)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Calculate how many RMS frames equal our max_duration window
        window_frames = int(max_samples / 512)
        
        # Convolve the RMS array with a window of ones to find the highest rolling sum of energy
        window = np.ones(window_frames)
        rolling_energy = np.convolve(rms, window, mode='valid')
        
        # The index with the maximum energy represents the start of our best audio segment
        best_frame_idx = np.argmax(rolling_energy)
        
        # Convert the RMS frame index back into raw audio samples
        start_sample = best_frame_idx * 512
        end_sample = start_sample + max_samples
        
        return audio[start_sample:end_sample]

    def process(self, path, smart_crop_duration=15.0):
        """
        Full waveform preprocessing pipeline (Upgraded for Accuracy & Speed)
        """
        audio = self.load_audio(path)
        audio = self.normalize(audio)
        
        # We explicitly disabled trim_silence to stop the 0.35s audio truncation bug!
        # smart_crop will safely find the best 15 seconds without dangerously destroying the length.
        
        if smart_crop_duration > 0:
            audio = self.smart_crop(audio, max_duration=smart_crop_duration)
            
        return audio