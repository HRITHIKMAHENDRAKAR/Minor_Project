# mock_main.py
import sys
import numpy as np

# Mock audio-separator for local test
class MockSeparator:
    def __init__(self, log_level="WARNING"): pass
    def load_model(self, model_filename): pass
    def separate(self, path): return ["mock1.wav", "mock2.wav"]

sys.modules['audio_separator'] = type('Mock', (), {})()
sys.modules['audio_separator.separator'] = type('MockSep', (), {'Separator': MockSeparator})

from preprocess.dataset_builder import DatasetBuilder
from preprocess.audio_separator import AudioSeparatorWrapper
import soundfile as sf

# Patch AudioSeparatorWrapper's separate method to bypass actual file writing for the mock stems
original_separate = AudioSeparatorWrapper.separate
def mock_separate(self, audio_array):
    # Just return 2 copies of the audio to simulate 2 stems
    return [audio_array, audio_array * 0.5]

AudioSeparatorWrapper.separate = mock_separate

def main():
    INPUT_DIR = "input"
    OUTPUT_DIR = "processed"

    print("Running mocked pipeline...")
    builder = DatasetBuilder()
    builder.build(INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
