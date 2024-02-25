"""
A Short script for splitting a dataset into Train, Validation and Test folders.
Because the dataset is very small, careful choseing of the split ratio is advised.

The following gives an even split between the Test and Validation sets:
    Train ratio: 0.78 - num_samples: 1945
    Validation ratio: 0.12 - num_samples: 288
    Test ratio: 0.1 - num_samples: 290 
"""

import argparse
import random
import shutil
from pathlib import Path

PICTURE_FILE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.tif']

def main(
        input_path: str,
        output_path: str,
        train_ratio: float = 0.78,
        val_ratio: float = 0.12,
        test_ratio: float = 0.1
    ):

    if (train_ratio + val_ratio + test_ratio) != 1.0:
        raise ValueError("The input ratios sum must be equal to 1!")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = {'train': {}, 'validation': {}, 'test': {}}
    for split in splits.keys():
        path = output_path / split
        path.mkdir(exist_ok=True)

    classes = [d for d in input_path.iterdir() if d.is_dir()]
    
    for class_dir in classes:
        # Get list of images for the current class
        images = [pic for pic in class_dir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]
        random.shuffle(images)
        
        # Calculate number of images for each split
        num_images = len(images)
        num_train = int(train_ratio * num_images)
        num_val = int(val_ratio * num_images)
        
        splits['train'][class_dir.name] = images[:num_train]
        splits['validation'][class_dir.name] = images[num_train:num_train + num_val] if test_ratio > 0.0 else images[num_train:]
        splits['test'][class_dir.name] = images[num_train + num_val:] if test_ratio > 0.0 else []

        for key, split in splits.items():
            for img in split[class_dir.name]:
                path = output_path / key / class_dir.name
                path.mkdir(exist_ok=True, parents=True)
                shutil.copy(img, path)

    #Print summary stat for the split to make sure the ratios given as argument worked as intended
    print(
        f"Train ratio: {train_ratio} - num_samples: {sum([len(cls) for cls in splits['train'].values()])}\n",
        f"Validation ratio: {val_ratio} - num_samples: {sum([len(cls) for cls in splits['validation'].values()])}\n",
        f"Test ratio: {test_ratio} - num_samples: {sum([len(cls) for cls in splits['test'].values()])}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="D:/UNI/PTE/Pollen/PollenDB/POLLEN73S_SEG_BG/SegmentedPollens")
    parser.add_argument("--output_path", type=str, default="D:/UNI/PTE/Pollen/PollenDB/POLLEN73S_Segmented_Pollen_Split")
    parser.add_argument("--train_ratio", type=float, default=0.78)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    main(input_path=args.input_path,
         output_path=args.output_path,
         train_ratio=args.train_ratio,
         val_ratio=args.val_ratio,
         test_ratio=args.test_ratio
         )