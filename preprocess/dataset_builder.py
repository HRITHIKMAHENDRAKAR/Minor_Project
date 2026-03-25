# preprocess/dataset_builder.py

import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .audio_processor import AudioProcessor
from .feature_extractor import FeatureExtractor
from .audio_separator import AudioSeparatorWrapper
from .config import TRAIN_SPLIT, VAL_SPLIT
from utils.file_utils import create_dir, get_species_folders


class DatasetBuilder:

    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.audio_separator = AudioSeparatorWrapper()
        self.feature_extractor = FeatureExtractor()

    def build(self, input_dir, output_dir):

        print("Starting dataset preprocessing...")

        species_folders = get_species_folders(input_dir)

        for species in species_folders:

            species_path = os.path.join(input_dir, species)
            files = [
                f for f in os.listdir(species_path)
                if f.endswith(".mp3")
            ]

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

                    # 1. Apply base waveform preprocessing (resample, normalize, trim, pad)
                    audio = self.audio_processor.process(file_path)
                    
                    # 2. Separate into multiple stems (e.g. background noise vs bird call, or bird 1 vs bird 2)
                    separated_stems = self.audio_separator.separate(audio)

                    for idx, stem_audio in enumerate(separated_stems):
                        # 3. Extract features for each stem
                        mel = self.feature_extractor.waveform_to_mel(stem_audio)

                        # 4. Save with an appended source index
                        output_name = file.replace(".mp3", f"_source_{idx}.npy")
                        output_path = os.path.join(split_species_dir, output_name)

                        np.save(output_path, mel)

        print("Dataset preprocessing completed successfully!")