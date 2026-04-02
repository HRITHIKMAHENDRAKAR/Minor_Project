# main.py

import sys
from preprocess.dataset_builder import DatasetBuilder
import eval_prebuilt

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sources", type=int, default=2)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--week", type=int, default=-1)
    parser.add_argument("--run_id", type=str, default="None")
    args = parser.parse_args()

    INPUT_DIR = "input"
    OUTPUT_DIR = "processed"
    
    n_sources = args.n_sources

    print("=== PHASE 1: Data Preparation & Source Separation ===")
    builder = DatasetBuilder(n_sources=n_sources)
    builder.build(INPUT_DIR, OUTPUT_DIR)
    from eval_prebuilt import main as eval_main
    print("\n=== PHASE 2: Inference & Evaluation (Prebuilt Models) ===")
    eval_main(lat=args.lat, lon=args.lon, week=args.week, n_sources=args.n_sources, run_id=args.run_id)


if __name__ == "__main__":
    main()