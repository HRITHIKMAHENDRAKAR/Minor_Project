# preprocess/dataset_builder.py

import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .audio_processor import AudioProcessor
from .feature_extractor import FeatureExtractor
from .audio_separator import AudioSeparatorWrapper
from .config import TRAIN_SPLIT, VAL_SPLIT, SAMPLE_RATE
from utils.file_utils import create_dir, get_species_folders


class DatasetBuilder:

    def __init__(self, n_sources=2):
        self.n_sources = n_sources
        self.audio_processor = AudioProcessor()
        self.audio_separator = AudioSeparatorWrapper(n_sources=n_sources)
        self.feature_extractor = FeatureExtractor()

    def build(self, input_dir, output_dir):

        print("Starting dataset preprocessing...")

        species_folders = get_species_folders(input_dir)

        for species in species_folders:

            species_path = os.path.join(input_dir, species)
            files = [
                f for f in os.listdir(species_path)
                if f.endswith((".mp3", ".wav", ".flac", ".ogg"))
            ]

            if len(files) < 3:
                # If there are not enough files for splitting (e.g., UI single upload), put everything in 'test'
                train_files, val_files, test_files = [], [], files
            else:
                train_files, temp_files = train_test_split(
                    files,
                    train_size=TRAIN_SPLIT,
                    random_state=42
                )
                val_files, test_files = train_test_split(
                    temp_files,
                    test_size=0.5,
                    random_state=42
                )

            split_map = {
                "train": train_files,
                "val": val_files,
                "test": test_files
            }

            for split, file_list in split_map.items():

                split_species_dir = os.path.join(output_dir, split, species)
                create_dir(split_species_dir)

                for file in tqdm(file_list, desc=f"{species} - {split}"):

                    file_path = os.path.join(species_path, file)

                    # 1. Apply base waveform preprocessing (resample, normalize, High-Pass, etc)
                    # CRITICAL: If hunting multiple birds, bypass the 15s smart_crop to process the full length!
                    crop_dur = 15.0 if self.n_sources <= 1 else 0.0
                    audio = self.audio_processor.process(file_path, smart_crop_duration=crop_dur)
                    
                    base_ext_stripped = os.path.splitext(file)[0]
                    # 2. Save the preprocessed original mixed audio (WAV + Spectrogram NPY)
                    orig_name_wav = f"{base_ext_stripped}_original.wav"
                    orig_name_npy = f"{base_ext_stripped}_original.npy"
                    
                    sf.write(os.path.join(split_species_dir, orig_name_wav), audio, SAMPLE_RATE)
                    np.save(os.path.join(split_species_dir, orig_name_npy), self.feature_extractor.waveform_to_mel(audio))

                    # 3. Separate into multiple stems (e.g. background noise vs bird call, or bird 1 vs bird 2)
                    separated_stems = self.audio_separator.separate(audio)

                    for idx, stem_audio in enumerate(separated_stems):
                        # 4. Save each separated stem (WAV + Spectrogram NPY)
                        stem_name_wav = f"{base_ext_stripped}_source_{idx}.wav"
                        stem_name_npy = f"{base_ext_stripped}_source_{idx}.npy"
                        
                        sf.write(os.path.join(split_species_dir, stem_name_wav), stem_audio, SAMPLE_RATE)
                        np.save(os.path.join(split_species_dir, stem_name_npy), self.feature_extractor.waveform_to_mel(stem_audio))

        print("Dataset preprocessing completed successfully!")