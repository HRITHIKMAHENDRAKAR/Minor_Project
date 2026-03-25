# utils/file_utils.py

import os


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_species_folders(input_dir):
    return [
        folder for folder in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, folder))
    ]