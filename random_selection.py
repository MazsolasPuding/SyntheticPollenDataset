import math
import random
import argparse
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
from PIL import Image

from Filters.complex_watershed_gpt import *
from Filters.complex_watershed_with_hsv import *
from Analyze_tools.hsv_histogram import *
from Analyze_tools.averaged_hsv_histogram import *

PICTURE_FILE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.tif']


def _one_per_class(path):
    selection = []
    for subdir in path.iterdir():
            if not subdir.is_dir():
                continue
            pictures = [pic for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]
            selection.append(random.choice(pictures))
    return selection

def _all_from_one_class(path):
    subdir = random.choice(list(path.iterdir()))
    return [pic for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]

def _n_random(path, num_samples):
    if num_samples == 0:
        num_samples = sum(1 for cls in path.iterdir() if cls.is_dir())

    all_pictures = [pic for subdir in path.iterdir() if subdir.is_dir()
                    for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]
    
    return random.choices(all_pictures, k=num_samples)


def plot_images(pictures):
    # Calculate rows and columns
    num_images = len(pictures.keys())
    cols = int(math.ceil(math.sqrt(num_images) + 2))
    rows = int(math.ceil(num_images / cols))

    # Create figure with dynamic size based on the number of images
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.5))
    axes = axes.flatten()  # Flatten to easily loop over it

    for ax, img_path in zip(axes, pictures.keys()):
        img = pictures[img_path]
        ax.imshow(img)
        ax.set_title(img_path.name, fontsize=8, y=-0.15)
        ax.axis('off')  # Hide axes ticks

    # Hide any unused subplots
    for ax in axes[len(pictures.keys()):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main(path: Path, mode: str, num_samples: int, seed: int = None):
    path = Path(path)
    if seed:
        random.seed(seed)
    
    if mode == 'one_per_class':
        selection = _one_per_class(path)
    elif mode == 'all_from_one_class':
        selection = _all_from_one_class(path)
    elif mode == 'n_random':
        selection = _n_random(path, num_samples)

    # pprint(selection)
    print(f"Number of selected pictures: {len(selection)}")

    pictures = {path : Image.open(path) for path in selection}
    for key in pictures.keys():
        pictures[key] = complex_watershed(pictures[key])
    # pprint(pictures.keys())
    plot_images(pictures)
    # avg_hist_hue, avg_hist_saturation, avg_hist_value = calculate_average_histogram(list(pictures.keys()))
    # plot_histograms_separate([avg_hist_hue, avg_hist_saturation, avg_hist_value], "Average HSV Histograms")



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='random_selection',
                                     description='Selects a random number of pollen samples from the DB')
    parser.add_argument('--path', help='DB Path', required=False, type=str, default='D:/UNI/PTE/Pollen/PollenDB/POLLEN73S')
    parser.add_argument('mode', choices=['one_per_class', 'all_from_one_class', 'n_random'], default='one_per_class',
                        help='Selection mode: one sample per class, N samples from one class, or totally random.')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of samples to select (applicable for all_from_one_class and random modes).')
    parser.add_argument('--seed', help='Random seed', type=int, default=42)
    args = parser.parse_args()

    main(path=args.path,
         mode=args.mode,
         num_samples=args.num_samples,
         seed=args.seed)
    