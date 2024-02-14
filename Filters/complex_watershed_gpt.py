import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def complex_watershed(image):
    # Read the image
    # image = cv2.imread(image_path)
    # Convert the PIL image to a numpy array (PIL to numpy)
    # image = np.array(image)
    # # Convert RGB (PIL) to BGR (OpenCV)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding after Gaussian filtering
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to remove small noises and holes within the pollen
    # Opening (erosion followed by dilation)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area (dilating to catch sure background)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area using distance transform and applying a suitable threshold.
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply the Watershed algorithm
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Optional: mark boundaries in red

    # Convert markers to binary image
    binary_image = np.uint8(markers == 1) * 255

    # Closing to ensure closed shapes, fill any holes
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Where mask is 0, set the image to be transparent
    result = cv2.bitwise_and(image, image, mask=closing)

    # Convert the original image to BGRA (so we have an alpha channel)
    result_bgra = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)

    # Set the alpha channel to the inverse of the mask
    result_bgra[:, :, 3] = cv2.bitwise_not(closing)

    # Convert the image to RGB
    rgba_image = cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2RGBA)
    return rgba_image


if __name__ == '__main__':
    # Provide the path to your image
    image_path = 'D:/UNI/PTE/Pollen/PollenDB/POLLEN73S/ceiba_speciosa/Figura24.TIF'
    image = cv2.imread(image_path)
    rgba_result = complex_watershed(image)

    # Display the original and the processed image
    image_np = np.array(image)
    plt.imshow(image_np)
    # Convert RGB (PIL) to BGR (OpenCV)
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    bgra_image = cv2.cvtColor(rgba_result, cv2.COLOR_RGBA2BGRA)
    plt.imshow(rgba_result)
    plt.show()

    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Pollen', bgra_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()