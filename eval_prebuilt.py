import os
import json
import pandas as pd
from tqdm import tqdm
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

def main(lat=None, lon=None, week=-1, n_sources=2):
    print("Loading BirdNET Analyzer... (This might take a moment to download the model if it's the first time)")
    analyzer = Analyzer()
    print("BirdNET loaded!")

    PROCESSED_DIR = "processed"
    
    # Store our evaluation results
    results = []

    # Target species list (scientific to common mapping is not strictly required, 
    # but we can just search for the scientific name in BirdNET's output)
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        split_dir = os.path.join(PROCESSED_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        species_folders = os.listdir(split_dir)
        
        for species in species_folders:
            # Species name has an underscore (e.g. "Corvus_splendens"), replace it to match BirdNET's scientific names
            species_clean = species.replace("_", " ")
            species_dir = os.path.join(split_dir, species)
            
            # Find all the original files in this species folder
            all_files = os.listdir(species_dir)
            original_wavs = [f for f in all_files if f.endswith("_original.wav")]
            
            print(f"Evaluating {len(original_wavs)} files for {species_clean} in {split} split...")
            
            for orig_file in tqdm(original_wavs):
                base_name = orig_file.replace("_original.wav", "")
                
                # Dynamically collect all ensemble files (Original + any Source stems generated)
                file_paths = {"Original": os.path.join(species_dir, orig_file)}
                i = 0
                while True:
                    source_path = os.path.join(species_dir, f"{base_name}_source_{i}.wav")
                    if os.path.exists(source_path):
                        file_paths[f"Source_{i}"] = source_path
                        i += 1
                    else:
                        break
                
                # We will record the highest confidence score BirdNET gives for our target species
                best_scores = {k: 0.0 for k in file_paths.keys()}
                
                # Dictionary to store accumulated scores for each species across all chunks and stems
                species_temporal_scores = {}
                species_max_confidence = {}
                
                overall_top_species = "None"
                overall_top_score_any = 0.0
                
                for key, path in file_paths.items():
                    if not os.path.exists(path):
                        continue
                        
                    # Initialize recording with a hyper-sensitive background threshold for multi-species Overlap Detection
                    recording = Recording(
                        analyzer,
                        path,
                        lat=lat,
                        lon=lon,
                        week_48=week,
                        min_conf=0.03, # Lowered from 0.1 to brilliantly catch faint/secondary birds!
                    )
                    recording.analyze()
                    
                    # Search through BirdNET's detections
                    target_score_for_stem = 0.0
                    for detection in recording.detections:
                        species = detection['scientific_name']
                        conf = detection['confidence']
                        
                        # Temporal Voting: Sum Confidences across chunks and sources
                        species_temporal_scores[species] = species_temporal_scores.get(species, 0.0) + conf
                        
                        # Track Max Confidence for final display
                        if conf > species_max_confidence.get(species, 0.0):
                            species_max_confidence[species] = conf
                            
                        # Legacy target tracking
                        if species.lower() == species_clean.lower():
                            if conf > target_score_for_stem:
                                target_score_for_stem = conf
                                
                    best_scores[key] = target_score_for_stem
                
                # Voting Logic Applied! Find the top N species with the HIGHEST TEMPORAL SCORES
                top_predictions = []
                if species_temporal_scores:
                    sorted_species = sorted(species_temporal_scores.items(), key=lambda item: item[1], reverse=True)
                    for species, temp_score in sorted_species:
                        conf = species_max_confidence[species]
                        top_predictions.append({"species": species, "confidence": float(conf)})
                        if len(top_predictions) >= max(1, n_sources):
                            break
                
                overall_top_species = top_predictions[0]["species"] if top_predictions else "None"
                overall_top_score_any = top_predictions[0]["confidence"] if top_predictions else 0.0
                
                # Determine the ensemble prediction (Max across all three for the target)
                ensemble_score = max(best_scores.values()) if best_scores else 0.0
                
                # Was the overall top prediction correct?
                is_correct = "Yes" if overall_top_species.lower() == species_clean.lower() else "No"
                
                res_dict = {
                    "Filename": base_name,
                    "True_Species": species_clean,
                    "Top_Predicted_Species": overall_top_species,
                    "Top_Prediction_Confidence": overall_top_score_any,
                    "Top_Predictions_JSON": json.dumps(top_predictions),
                    "Is_Prediction_Correct?": is_correct,
                    "Split": split,
                    "Score_Original": best_scores.get("Original", 0.0),
                }
                
                # Dynamically add source scores
                for idx in range(i):
                    res_dict[f"Score_Source{idx}"] = best_scores.get(f"Source_{idx}", 0.0)
                    
                res_dict["Ensemble_Score"] = ensemble_score
                res_dict["Did_Separation_Help?"] = "Yes" if ensemble_score > best_scores.get("Original", 0.0) else "No"
                
                results.append(res_dict)

    df_new = pd.DataFrame(results)
    csv_path = "prebuilt_birdnet_evaluation.csv"
    
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            # Combine old and new, keeping all dynamic columns safely
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
            df = df_combined # for the print statement below
        except Exception as e:
            print("Warning: Could not append to existing CSV. Overwriting...", e)
            df_new.to_csv(csv_path, index=False)
            df = df_new
    else:
        df_new.to_csv(csv_path, index=False)
        df = df_new
    
    print("\n--- INFERENCE RESULTS OVERVIEW ---")
    print(f"Total files evaluated: {len(df)}")
    
    # Calculate how many times the separation effectively boosted the confidence natively over the mixed file
    if "Did_Separation_Help?" in df.columns:
        helped_count = len(df[df["Did_Separation_Help?"] == "Yes"])
        print(f"Separation enhanced confidence scores in {helped_count} out of {len(df)} recordings ({(helped_count/len(df))*100:.1f}%)!")
    else:
        print("No evaluations were recorded.")
        
    print("Full report saved to 'prebuilt_birdnet_evaluation.csv'")


if __name__ == "__main__":
    main()
