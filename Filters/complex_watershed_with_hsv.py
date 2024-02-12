import cv2
import numpy as np
from PIL import Image

def complex_watershed_with_hsv(image):
    # # Read the image
    # image = cv2.imread(image_path)
    image_np = np.array(image)
    # Convert RGB (PIL) to BGR (OpenCV)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Convert to HSV color space
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for pollen (example values; adjust based on your dataset)
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([179, 255, 180])
    
    # Threshold the HSV image to get only pollen colors
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Apply Gaussian Blur to the mask to reduce noise
    blurred_mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Optional: Combine with Otsu's thresholding for grayscale image to refine the mask
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined_mask = cv2.bitwise_and(blurred_mask, otsu_thresh)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Apply the Watershed algorithm
    cv2.watershed(image_cv, markers)
    image_cv[markers == -1] = [255, 0, 0]  # Optional: mark boundaries in red
    
    # Convert markers to binary image for the pollen
    return np.uint8(markers == 1) * 255
    # binary_image = np.uint8(markers == 1) * 255
    
    # # Display the original and the processed image
    # cv2.imshow('Original Image', image_cv)
    # cv2.imshow('HSV Thresholded', combined_mask)
    # cv2.imshow('Segmented Pollen with HSV', binary_image)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    # Provide the path to your image
    image_path = 'D:/UNI/PTE/Pollen/PollenDB/POLLEN73S/ceiba_speciosa/Figura24.TIF'
    image = Image.open(image_path)
    complex_watershed_with_hsv(image)