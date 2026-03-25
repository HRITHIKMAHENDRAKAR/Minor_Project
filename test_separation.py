# test_separation.py

import os
import soundfile as sf
import numpy as np

from preprocess.audio_processor import AudioProcessor
from preprocess.audio_separator import AudioSeparatorWrapper

def main():
    # Attempt to find two distinct bird audio files to mix
    input_dir = "input"
    bird_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp3"):
                bird_files.append(os.path.join(root, file))
                if len(bird_files) == 2:
                    break
        if len(bird_files) == 2:
            break
            
    if len(bird_files) < 2:
        print("Not enough bird files found in input/ to mix.")
        return

    print(f"Mixing {bird_files[0]} and {bird_files[1]}")
    processor = AudioProcessor()
    
    # Process both audio files (resample, normalize, trim silence, pad/truncate to 5s)
    audio1 = processor.process(bird_files[0])
    audio2 = processor.process(bird_files[1])
    
    # Mix the audios
    mixed_audio = audio1 + audio2
    
    # Normalize the mixed audio to prevent clipping
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio / max_val
        
    print("Initializing separator...")
    separator = AudioSeparatorWrapper()
    
    print("Separating mixed audio...")
    stems = separator.separate(mixed_audio)
    
    print(f"Obtained {len(stems)} separated stems.")
    
    # Save the stems to disk for manual inspection
    os.makedirs("test_outputs", exist_ok=True)
    for i, stem in enumerate(stems):
        out_path = f"test_outputs/stem_{i}.wav"
        sf.write(out_path, stem, 22050)
        print(f"Saved {out_path}")
        
    # Also save the original mixture for comparison
    sf.write("test_outputs/mixed.wav", mixed_audio, 22050)
    print("Saved test_outputs/mixed.wav")

if __name__ == "__main__":
    main()
