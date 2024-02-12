import cv2
import numpy as np

# Path to the image you want to open
image_path = 'D:/UNI/PTE/Pollen/PollenDB/POLLEN73S/ceiba_speciosa/Figura24.TIF'

# Read the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
# The parameters can be adjusted to suit your image
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

_, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Stack both images for displaying them side by side
# Ensure both images are in BGR format if the original image is in color
combined = np.hstack((cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)))

# Display the original and thresholded image
cv2.imshow('Original (left) and Adaptive Threshold (right)', combined)

# Wait for a key press and then close all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()