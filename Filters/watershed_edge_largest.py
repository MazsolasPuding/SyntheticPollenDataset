import cv2
import numpy as np

def segment_pollen_with_edges(image):
    # Read the image
    # image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Enhance edges in the original grayscale image
    enhanced_gray = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)

    # Apply Otsu's thresholding on the enhanced grayscale image
    _, thresh = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply the Watershed algorithm
    markers = cv2.watershed(image, markers)

    # Create a binary image from the Watershed output
    segmented = np.zeros_like(gray)
    segmented[markers == 1] = 255

    # Find contours
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest external contour in the image is the pollen
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Create a new clean image to draw the largest contour filled
        clean_segmented = np.zeros_like(segmented)
        cv2.drawContours(clean_segmented, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return clean_segmented

        # Display the original and the processed image
        # cv2.imshow('Original Image', image)
        # cv2.imshow('Segmented Pollen', clean_segmented)
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("No contours found.")
        return segmented

if __name__ == '__main__':
    # Provide the path to your image
    image_path = 'path_to_your_pollen_image.jpg'
    segment_pollen_with_edges(image_path)
