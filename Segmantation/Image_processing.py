import math
import random
import argparse
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2

# from Filters import *
from Analyze_tools import *
from Filters.watershed_edge import segment_pollen_with_edges
from Filters.watershed_edge_largest import segment_pollen_with_edges
from Filters.complex_watershed_gpt import complex_watershed
from Segmentation_Using_AI.apply_rembg import apply_rembg_remove


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
    subdir = random.choice([sub_path for sub_path in path.iterdir() if sub_path.is_dir()])
    return [pic for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]

def _n_random(path, num_samples):
    if num_samples == 0:
        num_samples = sum(1 for cls in path.iterdir() if cls.is_dir())

    all_pictures = [pic for subdir in path.iterdir() if subdir.is_dir()         # Get all Images in DB
                    for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]
    
    return random.choices(all_pictures, k=num_samples)

def _all(path):
    return  [pic for subdir in path.iterdir() if subdir.is_dir()         # Get all Images in DB
                    for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]

def image_shape_check(pictures):
    for key in pictures.keys():
        # Check image mode before processing
        print(f"Input type: {type(pictures[key])}, shape: {np.array(pictures[key]).shape}")

        rgba_result = complex_watershed(pictures[key])

        # Check the result's type and shape
        print(f"Result type: {type(rgba_result)}, shape: {np.array(rgba_result).shape}")

        # If the result is a PIL Image, convert it to an array for imshow
        if isinstance(rgba_result, Image.Image):
            rgba_result = np.array(rgba_result)

        # Ensure that the result is in the correct format (RGBA) for imshow
        if rgba_result.shape[2] == 4:
            bgra_image = cv2.cvtColor(rgba_result, cv2.COLOR_RGBA2BGRA)
            cv2.imshow('', bgra_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("The result does not have 4 channels. It may be displayed incorrectly.")

        # plt.axis('off')
        # plt.show()

def create_directory_tree(input_path: Path, save_path: Path, keep_bg: bool = False):
    save_path.mkdir(exist_ok=True, parents=True)
    if keep_bg:
        paths = [save_path / "SegmentedPollens", save_path / "SegmentedBackground"]
        for path in paths:
            for subfolder in [sub for sub in input_path.iterdir() if sub.is_dir()]:
                sub_path = path / subfolder.name
                sub_path.mkdir(exist_ok=True, parents=True)
                print(f"Created: {sub_path}")
    else:
        for subfolder in [sub for sub in input_path.iterdir() if sub.is_dir()]:
            sub_path = save_path / subfolder.name
            sub_path.mkdir(exist_ok=True, parents=True)
            print(f"Created: {sub_path}")

def plot_images(pictures: np.array):
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


def main(path: Path,
         mode: str,
         num_samples: int,
         seed: int = None,
         preprocess: bool = False,
         save_path: str = None,
         keep_bg: bool = False,
         plot_selection: bool = False,
         plot_analytics: bool = False):
    
    # Setup environment and inputs
    path = Path(path)
    if save_path: save_path = Path(save_path)
    if seed: random.seed(seed)

    # Selecting Random Images from DB
    if mode == 'one_per_class':
        selection = _one_per_class(path)
    elif mode == 'all_from_one_class':
        selection = _all_from_one_class(path)
    elif mode == 'n_random':
        selection = _n_random(path, num_samples)
    elif mode == 'all':
        selection = _all(path)
    print(f"Number of selected pictures: {len(selection)}")


    # Read Selection
    input_images = {path : cv2.imread(str(path)) for path in selection}

    # Analytics of input Images
    if plot_analytics:
        show_classe_distribution(path)
        show_size_distribution(path)
        avg_hist_hue, avg_hist_saturation, avg_hist_value = calculate_average_histogram(list(input_images.keys()))
        plot_histograms_separate([avg_hist_hue, avg_hist_saturation, avg_hist_value], "Average HSV Histograms")

    if save_path:
        create_directory_tree(input_path=path, save_path=save_path, keep_bg=keep_bg)

    # Image Processing
    output_images = {}
    if preprocess:
        for key in tqdm(input_images.keys()):
            # rgba_result = complex_watershed(input_images[key])
            # rgba_result = segment_pollen_with_edges(input_images[key])
            if not save_path:
                rgba_result = apply_rembg_remove(image=input_images[key],
                                                 return_img=plot_selection,
                                                 keep_bg=keep_bg)
                output_images[key] = rgba_result
            else:
                if keep_bg:
                    img_save_path = str(save_path / "SegmentedPollens" / key.parent.name / key.stem) + ".png"
                    bg_save_path =  str(save_path / "SegmentedBackground" / key.parent.name / key.stem) + ".png"
                else:
                    img_save_path = str(save_path / key.parent.name / key.stem) + ".png"
                    bg_save_path =  None

                apply_rembg_remove(image=input_images[key],
                                   return_img=plot_selection,
                                   keep_bg=keep_bg,
                                   image_save_path=img_save_path,
                                   bg_save_path=bg_save_path)
    else:
        output_images = input_images

    # Debug
    # image_shape_check(input_images)

    if plot_selection:
        plot_images(output_images)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='random_selection',
                                     description='Selects a random number of pollen samples from the DB')
    parser.add_argument('--path', help='DB Path', required=False, type=str, default='D:/UNI/PTE/Pollen/PollenDB/POLLEN73S_SEG_BG/SegmentedBackground')
    parser.add_argument('--mode', choices=['one_per_class', 'all_from_one_class', 'n_random', 'all'], default='one_per_class',
                        help='Selection mode: one sample per class, N samples from one class, or totally random.')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of samples to select (applicable for all_from_one_class and random modes).')
    parser.add_argument('--seed', help='Random seed', type=int, default=42)
    parser.add_argument('--preprocess', help="Apply Filters and Preprocesses to Selected Images", type=bool, default=True)
    parser.add_argument('--save_path', help="Save path for Preprocessed Images", type=str, default=None)
    parser.add_argument('--keep_bg', help="Keep Background of pollen images and save them", type=bool, default=True)
    parser.add_argument('--plot_selection', help="Plot images of selected pollens", type=bool, default=True)
    parser.add_argument('--plot_analytics', help="Plot analytics of selection", type=bool, default=True)
    args = parser.parse_args()

    main(path=args.path,
         mode=args.mode,
         num_samples=args.num_samples,
         seed=args.seed,
         preprocess=args.preprocess,
         save_path=args.save_path,
         keep_bg=args.keep_bg,
         plot_selection=args.plot_selection,
         plot_analytics=args.plot_analytics)
    
    # 'D:/UNI/PTE/Pollen/Classification/data/KaggleDB_Structured'
    # 'D:/UNI/PTE/Pollen/PollenDB/POLLEN73S'
    # 'D:/UNI/PTE/Pollen/PollenDB/POLLEN73S/hyptis_sp/hyptis_sp (35).jpg'
    # 'D:/UNI/PTE/Pollen/PollenDB/POLLEN73S_SEG'
    # /Users/horvada/Git/Personal/PollenDB/POLLEN73S
    # /Users/horvada/Git/Personal/PollenDB/KaggleDB_Structured
