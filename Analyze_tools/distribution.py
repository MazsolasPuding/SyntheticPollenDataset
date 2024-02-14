from pathlib import Path

import cv2
import matplotlib.pyplot as plt

PICTURE_FILE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.tif']

def show_classe_distribution(path):
    classes = {cls.name: len(list(cls.iterdir())) for cls in path.iterdir() if cls.is_dir()}
    print(classes)
    plt.figure(figsize = (12,8))
    plt.title('Class Counts in Dataset')
    plt.bar(*zip(*classes.items()))
    plt.xticks(rotation='vertical')
    plt.show()

def show_size_distribution(path):
    """Show scatter plot of image sizes."""
    all_images = [pic for subdir in path.iterdir() if subdir.is_dir()         # Get all Images in DB
                    for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]
    size = [cv2.imread(str(image_path)).shape for image_path in all_images]
    x, y, _ = zip(*size)
    fig = plt.figure(figsize=(12, 10))

    # scatter plot
    plt.scatter(x,y)
    plt.title("Image size scatterplot")

    # add diagonal red line 
    max_dim = max(max(x), max(y))
    plt.plot([0, max_dim],[0, max_dim], 'r')
    plt.show()

if __name__ == '__main__':
    show_classe_distribution(path=Path('D:/UNI/PTE/Pollen/PollenDB/POLLEN73S'))
    show_size_distribution(path=Path('D:/UNI/PTE/Pollen/PollenDB/POLLEN73S'))
    show_classe_distribution(path=Path('D:/UNI/PTE/Pollen/Classification/data/KaggleDB_Structured'))
    show_size_distribution(path=Path('D:/UNI/PTE/Pollen/Classification/data/KaggleDB_Structured'))