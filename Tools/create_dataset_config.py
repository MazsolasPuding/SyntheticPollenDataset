"""
Creates a dataset config yaml file for YOLO annotation format
ruamel.yaml library is used instead of the standard yaml dict,
because it supports key order preservation of the dictionary.
This is just to enhance readibility of the rsulting file.
"""

import argparse
from pathlib import Path
from ruamel.yaml import YAML


def create_dataset_config(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    classes = sorted([cls.name for cls in input_path.iterdir() if cls.is_dir()])
    classes_dict = {ind: cls for ind, cls in enumerate(classes)}

    data = {
        'path': str(output_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': classes_dict
    }

    with open(output_path / 'pollen_dataset.yaml', 'w') as file:
        yaml = YAML()
        yaml.dump(data, file)
    print(f"YAML data has been written to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Input dataset Path", type=str, default="D:/UNI/PTE/Pollen/datasets/POLLEN73S_SEG_BG/SegmentedPollens")
    parser.add_argument("--output_path", help="Output Yaml Path", type=str, default="D:/UNI/PTE/Pollen/datasets/SYNTH_dataset_POLLEN73S")
    args = parser.parse_args()

    create_dataset_config(input_path=args.input_path, output_path=args.output_path)