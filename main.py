# main.py

from preprocess.dataset_builder import DatasetBuilder


def main():

    INPUT_DIR = "input"
    OUTPUT_DIR = "processed"

    builder = DatasetBuilder()
    builder.build(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()