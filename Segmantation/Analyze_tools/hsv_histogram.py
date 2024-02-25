import io

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def hsv_histogram(image):
    # Load the image
    image_np = np.array(image)
    # Convert RGB (PIL) to BGR (OpenCV)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert the image from BGR to HSV
    image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    # Parameters for histogram
    # hue ranges from 0 to 180, saturation and value from 0 to 255
    hist_bins = [180, 256, 256]  # Number of bins for H, S, V
    hist_ranges = [0, 180, 0, 256, 0, 256]  # Ranges for H, S, V

    # Compute the histograms
    hist_hue = cv2.calcHist([image_hsv], [0], None, [hist_bins[0]], [hist_ranges[0], hist_ranges[1]])
    hist_saturation = cv2.calcHist([image_hsv], [1], None, [hist_bins[1]], [hist_ranges[2], hist_ranges[3]])
    hist_value = cv2.calcHist([image_hsv], [2], None, [hist_bins[2]], [hist_ranges[4], hist_ranges[5]])

    # Plotting
    plt.figure(figsize=(10, 8))

    plt.subplot(311)  # Hue
    plt.plot(hist_hue, color='red')
    plt.title('Hue')

    plt.subplot(312)  # Saturation
    plt.plot(hist_saturation, color='green')
    plt.title('Saturation')

    plt.subplot(313)  # Value
    plt.plot(hist_value, color='blue')
    plt.title('Value')

    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Rewind the buffer to the beginning so you can read from it

    # Create a PIL image from the buffer
    img = Image.open(buf)
    # Don't forget to close the buffer
    # buf.close()

    return img

if __name__ == '__main__':
    pic = hsv_histogram(image_path='D:/UNI/PTE/Pollen/PollenDB/POLLEN73S/ceiba_speciosa/Figura24.TIF').show()
    # plt.imshow(pic)
    